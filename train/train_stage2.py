"""Command-line orchestration for frozen-contributor weight regression."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from data.dataset import (
    DatasetIndex,
    FrameRecord,
    audit_records,
    load_dataset_index,
    load_rgbd_sample,
)
from data.preprocessing import PreprocessConfig, normalize_sample
from data.sampling import PigBalancedSampler
from model.two_stage_model import (
    STAGE1_CLASS_NAMES,
    CheckpointCompatibilityError,
    Stage1Metadata,
    TwoStagePigWeightModel,
    assert_compatible_metadata,
    load_frozen_contributor,
)
from model.vit_regressor import PigWeightViT
from train.common import (
    build_grad_scaler,
    build_optimizer_scheduler,
    collect_environment,
    seed_everything,
    seed_worker,
    select_device,
)
from train.config import load_config, validate_stage2_config
from train.run_io import RunDirectory
from train.stage2_engine import evaluate_stage2, train_stage2_epoch


_INPUT_CHANNELS = {"rgb": 3, "depth": 1, "rgbd": 4}
_SPLIT_NAMES = ("train", "validation", "cross_scene_test")
_STAGE2_SCHEMA_VERSION = 1


@dataclass(slots=True)
class Stage2Runtime:
    """All dependencies and provenance for one stage-two run."""

    config: dict[str, Any]
    run: RunDirectory
    model: TwoStagePigWeightModel
    stage1_metadata: Stage1Metadata
    stage1_checkpoint_path: Path
    stage1_checkpoint_sha256: str
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: Any | None
    smooth_l1_beta: float
    train_sampler: PigBalancedSampler
    train_loader: Any
    evaluation_loaders: Mapping[str, Any]
    device: torch.device


class _Stage2FrameDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Normalize original frames without constructing augmentation machinery."""

    def __init__(
        self, records: Sequence[FrameRecord], preprocess_config: PreprocessConfig,
        *, modalities: Sequence[str] | None = None,
    ) -> None:
        self.records = tuple(records)
        self.preprocess_config = preprocess_config
        self.modalities = None if modalities is None else tuple(dict.fromkeys(modalities))
        if self.modalities is not None and (
            not self.modalities or any(name not in _INPUT_CHANNELS for name in self.modalities)
        ):
            raise ValueError("modalities must contain supported stage-two inputs")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = normalize_sample(
            load_rgbd_sample(self.records[index]), self.preprocess_config
        )
        if self.modalities is None:
            return sample
        # Avoid collating/pinning unused RGB, depth, segmentation labels and metadata.
        return {name: sample[name] for name in (
            *self.modalities, "weight_kg", "split", "pig_id", "frame_id"
        )}


def _require_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate_runtime_config(config: Mapping[str, Any]) -> None:
    training = config["training"]
    _require_positive_integer("training.epochs", training["epochs"])
    _require_positive_integer("training.batch_size", training["batch_size"])
    _require_nonnegative_integer("training.num_workers", training["num_workers"])
    samples_per_epoch = config["data"]["samples_per_epoch"]
    if samples_per_epoch is not None:
        _require_positive_integer("data.samples_per_epoch", samples_per_epoch)
    if config["loss"]["name"] != "smooth_l1":
        raise ValueError("loss.name must be 'smooth_l1'")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preprocess_config(config: Mapping[str, Any]) -> PreprocessConfig:
    return PreprocessConfig(
        image_size=tuple(config["data"]["image_size"]),
        patch_size=config["data"]["patch_size"],
        depth_max_mm=config["data"]["depth_max_mm"],
    )


def _audit_index(
    index: DatasetIndex, invalid_sample_policy: str
) -> tuple[DatasetIndex, dict[str, Any]]:
    valid_by_split: dict[str, tuple[FrameRecord, ...]] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    for split_name in _SPLIT_NAMES:
        valid, report = audit_records(index.by_split[split_name], invalid_sample_policy)
        if not valid:
            raise ValueError(f"stage-two {split_name} split has no valid records")
        valid_by_split[split_name] = valid
        split_reports[split_name] = report
    return DatasetIndex(index.dataset_root, valid_by_split), {
        "dataset_root": str(index.dataset_root),
        "invalid_sample_policy": invalid_sample_policy,
        "input_count": sum(report["input_count"] for report in split_reports.values()),
        "valid_count": sum(report["valid_count"] for report in split_reports.values()),
        "skipped_count": sum(
            report["skipped_count"] for report in split_reports.values()
        ),
        "splits": split_reports,
    }


def _record_manifest(records: Sequence[FrameRecord]) -> dict[str, Any]:
    return {
        "pig_ids": sorted({record.pig_id for record in records}),
        "frame_count": len(records),
        "frames": [
            {
                "split": record.split,
                "scene": record.scene,
                "pig_id": record.pig_id,
                "frame_id": record.frame_id,
                "weight_kg": record.weight_kg,
                "rgb_path": str(record.rgb_path),
                "depth_path": str(record.depth_path),
                "depth_seg_path": str(record.depth_seg_path),
                "mask_path": str(record.mask_path),
            }
            for record in records
        ],
    }


def _validate_original_split_isolation(index: DatasetIndex) -> None:
    pig_splits: dict[str, str] = {}
    for split_name in _SPLIT_NAMES:
        for record in index.by_split[split_name]:
            prior_split = pig_splits.setdefault(record.pig_id, split_name)
            if prior_split != split_name:
                raise ValueError(
                    f"pig_id {record.pig_id!r} appears in multiple original splits: "
                    f"{prior_split!r} and {split_name!r}"
                )


def _make_loader(
    dataset: _Stage2FrameDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    sampler: PigBalancedSampler | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    if pin_memory is None:
        pin_memory = sys.platform != "win32"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda" and pin_memory,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def _validate_contributor_metadata(
    metadata: Stage1Metadata,
    config: Mapping[str, Any],
    preprocess: PreprocessConfig,
) -> None:
    if metadata.modality not in _INPUT_CHANNELS:
        raise CheckpointCompatibilityError(
            f"unsupported stage-1 contributor modality: {metadata.modality!r}"
        )
    expected = dataclasses.replace(
        metadata,
        input_channels=_INPUT_CHANNELS[metadata.modality],
        image_size=tuple(config["data"]["image_size"]),
        patch_size=config["data"]["patch_size"],
        class_names=STAGE1_CLASS_NAMES,
        depth_max_mm=preprocess.depth_max_mm,
        rgb_mean=preprocess.rgb_mean,
        rgb_std=preprocess.rgb_std,
    )
    assert_compatible_metadata(metadata, expected)


def build_runtime(config: dict[str, Any]) -> Stage2Runtime:
    """Load the required contributor and build loaders from the original splits."""
    validate_stage2_config(config)
    _validate_runtime_config(config)
    checkpoint_path = Path(config["stage1_checkpoint"]).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"stage1 checkpoint does not exist: {checkpoint_path}")

    resolved_config = copy.deepcopy(config)
    resolved_config["loss"]["beta"] = float(resolved_config["loss"]["beta"])
    resolved_config["model"].setdefault("expert_routing", False)
    resolved_config["model"].setdefault("background_mode", "drop")
    training = resolved_config["training"]
    training.setdefault("evaluation_batch_size", min(training["batch_size"], 8))
    training.setdefault("pin_memory", sys.platform != "win32")
    seed_everything(training["seed"])
    device = select_device(training["device"])
    preprocess = _preprocess_config(resolved_config)

    contributor, stage1_metadata, _ = load_frozen_contributor(
        checkpoint_path, expected=None, map_location=device
    )
    _validate_contributor_metadata(stage1_metadata, resolved_config, preprocess)
    contributor = contributor.to(device)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    resolved_config["stage1_checkpoint_resolved"] = str(checkpoint_path)
    resolved_config["stage1_checkpoint_sha256"] = checkpoint_sha256
    resolved_config["stage1_contributor_metadata"] = dataclasses.asdict(
        stage1_metadata
    )

    raw_index = load_dataset_index(resolved_config["data"]["split_json"])
    valid_index, audit_report = _audit_index(
        raw_index, resolved_config["data"]["invalid_sample_policy"]
    )
    _validate_original_split_isolation(valid_index)
    split_records = {
        split_name: valid_index.by_split[split_name] for split_name in _SPLIT_NAMES
    }

    run = RunDirectory.create(
        Path(training["runs_root"]),
        "stage2",
        resolved_config["model"]["regressor_modality"],
        resolved_config["model"]["regressor_profile"],
    )
    environment = collect_environment()
    environment["stage1_checkpoint_resolved"] = str(checkpoint_path)
    environment["stage1_checkpoint_sha256"] = checkpoint_sha256
    run.write_yaml("resolved_config.yaml", resolved_config)
    run.write_json("environment.json", environment)
    run.write_json("data_audit.json", audit_report)
    run.write_json(
        "split_manifest.json",
        {
            "splits": {
                split_name: _record_manifest(records)
                for split_name, records in split_records.items()
            }
        },
    )

    modalities = (stage1_metadata.modality, resolved_config["model"]["regressor_modality"])
    training_dataset = _Stage2FrameDataset(
        split_records["train"], preprocess, modalities=modalities
    )
    evaluation_datasets = {
        split_name: _Stage2FrameDataset(records, preprocess, modalities=modalities)
        for split_name, records in split_records.items()
    }
    samples_per_epoch = resolved_config["data"]["samples_per_epoch"]
    train_sampler = PigBalancedSampler(
        split_records["train"],
        num_samples=(
            len(split_records["train"])
            if samples_per_epoch is None
            else samples_per_epoch
        ),
        seed=training["seed"],
    )
    train_loader = _make_loader(
        training_dataset,
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
        device=device,
        seed=training["seed"],
        sampler=train_sampler,
        pin_memory=training["pin_memory"],
    )
    evaluation_loaders = {
        split_name: _make_loader(
            dataset,
            batch_size=training["evaluation_batch_size"],
            num_workers=training["num_workers"],
            device=device,
            seed=training["seed"] + index + 1,
            pin_memory=training["pin_memory"],
        )
        for index, (split_name, dataset) in enumerate(evaluation_datasets.items())
    }

    image_size = tuple(resolved_config["data"]["image_size"])
    patch_size = resolved_config["data"]["patch_size"]
    patch_count = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    print(f"Stage 2 patch tokens per frame: {patch_count}")
    regressor_modality = resolved_config["model"]["regressor_modality"]
    regressor = PigWeightViT(
        _INPUT_CHANNELS[regressor_modality],
        image_size,
        patch_size,
        resolved_config["model"]["regressor_profile"],
        overrides=resolved_config["model"]["regressor_overrides"],
        expert_routing=resolved_config["model"]["expert_routing"],
        background_mode=resolved_config["model"]["background_mode"],
    ).to(device)
    connectivity_settings = resolved_config.get("connectivity", {})
    model = TwoStagePigWeightModel(
        contributor, regressor,
        enforce_connectivity=connectivity_settings.get("postprocess", False),
        connectivity=connectivity_settings.get("neighborhood", 8),
    ).to(device)
    optimizer, scheduler, _ = build_optimizer_scheduler(
        model.regressor.parameters(),
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        epochs=training["epochs"],
        steps_per_epoch=len(train_loader),
        warmup_epochs=training["warmup_epochs"],
        patch_count=patch_count,
        profiles=(stage1_metadata.profile, resolved_config["model"]["regressor_profile"]),
    )
    scaler = build_grad_scaler(device, enabled=bool(training["amp"]))
    return Stage2Runtime(
        config=resolved_config,
        run=run,
        model=model,
        stage1_metadata=stage1_metadata,
        stage1_checkpoint_path=checkpoint_path,
        stage1_checkpoint_sha256=checkpoint_sha256,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        smooth_l1_beta=resolved_config["loss"]["beta"],
        train_sampler=train_sampler,
        train_loader=train_loader,
        evaluation_loaders=evaluation_loaders,
        device=device,
    )


def _dry_run_split_counts(records: Sequence[FrameRecord]) -> dict[str, int]:
    return {
        "frames": len(records),
        "pigs": len({record.pig_id for record in records}),
    }


def run_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    """Audit, load frozen stage one, and forward each loader without writing."""
    validate_stage2_config(config)
    _validate_runtime_config(config)
    checkpoint_path = Path(config["stage1_checkpoint"]).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"stage1 checkpoint does not exist: {checkpoint_path}")

    training = config["training"]
    seed_everything(training["seed"])
    device = select_device(training["device"])
    preprocess = _preprocess_config(config)
    contributor, stage1_metadata, _ = load_frozen_contributor(
        checkpoint_path, expected=None, map_location=device
    )
    _validate_contributor_metadata(stage1_metadata, config, preprocess)
    contributor = contributor.to(device)

    raw_index = load_dataset_index(config["data"]["split_json"])
    valid_index, audit_report = _audit_index(
        raw_index, config["data"]["invalid_sample_policy"]
    )
    _validate_original_split_isolation(valid_index)
    split_records = {
        split_name: valid_index.by_split[split_name] for split_name in _SPLIT_NAMES
    }
    modalities = (stage1_metadata.modality, config["model"]["regressor_modality"])
    loaders = {
        "optimization_train": _make_loader(
            _Stage2FrameDataset((split_records["train"][0],), preprocess, modalities=modalities),
            batch_size=1,
            num_workers=0,
            device=device,
            seed=training["seed"],
            pin_memory=training.get("pin_memory"),
        )
    }
    loaders.update(
        {
            split_name: _make_loader(
                _Stage2FrameDataset((records[0],), preprocess, modalities=modalities),
                batch_size=1,
                num_workers=0,
                device=device,
                seed=training["seed"] + index + 1,
                pin_memory=training.get("pin_memory"),
            )
            for index, (split_name, records) in enumerate(split_records.items())
        }
    )

    image_size = tuple(config["data"]["image_size"])
    patch_size = config["data"]["patch_size"]
    token_count = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    regressor_modality = config["model"]["regressor_modality"]
    regressor = PigWeightViT(
        _INPUT_CHANNELS[regressor_modality],
        image_size,
        patch_size,
        config["model"]["regressor_profile"],
        overrides=config["model"]["regressor_overrides"],
        expert_routing=config["model"].get("expert_routing", False),
        background_mode=config["model"].get("background_mode", "drop"),
    ).to(device)
    connectivity_settings = config.get("connectivity", {})
    model = TwoStagePigWeightModel(
        contributor, regressor,
        enforce_connectivity=connectivity_settings.get("postprocess", False),
        connectivity=connectivity_settings.get("neighborhood", 8),
    ).to(device)
    model.eval()
    trainable_contributor_parameters = sum(
        parameter.numel()
        for parameter in model.contributor.parameters()
        if parameter.requires_grad
    )
    contributor_frozen = (
        trainable_contributor_parameters == 0 and not model.contributor.training
    )
    if not contributor_frozen:
        raise RuntimeError("stage-two dry-run contributor is not strictly frozen")

    batches: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for loader_name, loader in loaders.items():
            batch = next(iter(loader))
            contributor_input = batch[stage1_metadata.modality].to(device)
            regressor_input = batch[regressor_modality].to(device)
            output = model(contributor_input, regressor_input)
            batches[loader_name] = {
                "contributor_input_shape": list(contributor_input.shape),
                "regressor_input_shape": list(regressor_input.shape),
                "output_shape": list(output.prediction.shape),
                "selected_counts": output.selected_counts.tolist(),
                "fallback_count": output.fallback_count,
            }

    report = {
        "stage": "stage2",
        "mode": "dry_run",
        "device": str(device),
        "image_size": list(image_size),
        "patch_size": patch_size,
        "token_count": token_count,
        "split_counts": {
            name: _dry_run_split_counts(records)
            for name, records in split_records.items()
        },
        "audit": {
            "input_count": audit_report["input_count"],
            "valid_count": audit_report["valid_count"],
            "skipped_count": audit_report["skipped_count"],
        },
        "stage1_checkpoint": str(checkpoint_path),
        "stage1_checkpoint_sha256": _sha256_file(checkpoint_path),
        "contributor_frozen": contributor_frozen,
        "trainable_contributor_parameters": trainable_contributor_parameters,
        "expert_routing": bool(getattr(model.regressor, "expert_routing", False)),
        "background_mode": getattr(model.regressor, "background_mode", "drop"),
        "trainable_expert_parameters": sum(
            parameter.numel()
            for name, parameter in model.regressor.named_parameters()
            if name.startswith("expert_router.") and parameter.requires_grad
        ),
        "batches": batches,
    }
    print(json.dumps(report, sort_keys=True))
    return report


def _checkpoint_payload(
    runtime: Stage2Runtime, epoch: int, best_validation_pig_mae: float
) -> dict[str, Any]:
    return {
        "schema_version": _STAGE2_SCHEMA_VERSION,
        "epoch": epoch,
        "regressor_state": runtime.model.regressor.state_dict(),
        "optimizer_state": runtime.optimizer.state_dict(),
        "scheduler_state": runtime.scheduler.state_dict(),
        "best_validation_pig_mae": best_validation_pig_mae,
        "loss": {
            "name": "smooth_l1",
            "beta": runtime.smooth_l1_beta,
        },
        "contribution_decoding": {
            "enforce_connectivity": runtime.model.enforce_connectivity,
            "connectivity": runtime.model.connectivity,
        },
        "stage1_checkpoint_resolved": str(runtime.stage1_checkpoint_path),
        "stage1_checkpoint_sha256": runtime.stage1_checkpoint_sha256,
        "regressor_metadata": {
            "modality": runtime.config["model"]["regressor_modality"],
            "profile": runtime.config["model"]["regressor_profile"],
            "overrides": copy.deepcopy(
                runtime.config["model"]["regressor_overrides"]
            ),
            "image_size": tuple(runtime.config["data"]["image_size"]),
            "patch_size": runtime.config["data"]["patch_size"],
            "expert_routing": bool(
                getattr(runtime.model.regressor, "expert_routing", False)
            ),
            "background_mode": getattr(runtime.model.regressor, "background_mode", "drop"),
        },
    }


def _print_report(epoch: int, split: str, report: Mapping[str, Any]) -> None:
    frame = report["frame"]
    pig = report["pig"]
    print(
        f"{epoch:5d} {split:16s} {float(report['loss']):10.5f} "
        f"{float(frame['mae']):10.5f} {float(frame['rmse']):10.5f} "
        f"{float(pig['mae']):10.5f} {float(pig['rmse']):10.5f} "
        f"{int(report['fallback_count']):9d}"
    )


def _release_unused_cuda_memory(device: torch.device) -> None:
    """Release unused device reservations between optimization and evaluation."""
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def run_training(runtime: Stage2Runtime) -> None:
    """Train the regressor, evaluate original splits, and export best predictions."""
    epochs = _require_positive_integer(
        "training.epochs", runtime.config["training"]["epochs"]
    )
    best_validation_pig_mae = math.inf
    print(
        "epoch split                  loss  frame_mae frame_rmse    pig_mae "
        "  pig_rmse fallbacks"
    )
    for epoch_index in range(epochs):
        epoch_number = epoch_index + 1
        runtime.train_sampler.set_epoch(epoch_index)
        train_stage2_epoch(
            runtime.model,
            runtime.train_loader,
            runtime.optimizer,
            runtime.device,
            contributor_modality=runtime.stage1_metadata.modality,
            regressor_modality=runtime.config["model"]["regressor_modality"],
            scaler=runtime.scaler,
            grad_clip=runtime.config["training"]["grad_clip"],
            scheduler=runtime.scheduler,
            beta=runtime.smooth_l1_beta,
            show_progress=True,
            progress_description=f"Stage 2 epoch {epoch_number}/{epochs}",
        )
        _release_unused_cuda_memory(runtime.device)

        reports: dict[str, dict[str, Any]] = {}
        for split_name in _SPLIT_NAMES:
            report, _ = evaluate_stage2(
                runtime.model,
                runtime.evaluation_loaders[split_name],
                runtime.device,
                contributor_modality=runtime.stage1_metadata.modality,
                regressor_modality=runtime.config["model"]["regressor_modality"],
                beta=runtime.smooth_l1_beta,
            )
            reports[split_name] = report
            runtime.run.append_metrics(epoch_number, split_name, report)
            _print_report(epoch_number, split_name, report)
            _release_unused_cuda_memory(runtime.device)

        validation_pig_mae = float(reports["validation"]["pig"]["mae"])
        is_best = validation_pig_mae < best_validation_pig_mae
        if is_best:
            best_validation_pig_mae = validation_pig_mae
        payload = _checkpoint_payload(
            runtime, epoch_number, best_validation_pig_mae
        )
        runtime.run.write_checkpoint("last.pt", payload)
        if is_best:
            runtime.run.write_checkpoint("best.pt", payload)

    best_payload = torch.load(
        runtime.run.checkpoint_path("best.pt"),
        map_location=runtime.device,
        weights_only=False,
    )
    runtime.model.regressor.load_state_dict(
        best_payload["regressor_state"], strict=True
    )
    for split_name in _SPLIT_NAMES:
        _, rows = evaluate_stage2(
            runtime.model,
            runtime.evaluation_loaders[split_name],
            runtime.device,
            contributor_modality=runtime.stage1_metadata.modality,
            regressor_modality=runtime.config["model"]["regressor_modality"],
            beta=runtime.smooth_l1_beta,
        )
        runtime.run.write_predictions(rows, f"{split_name}_frames.csv")
        runtime.run.write_pig_predictions(rows, f"{split_name}_pigs.csv")
        _release_unused_cuda_memory(runtime.device)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "stage2.yaml",
        help="Path to the stage-two YAML configuration.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing dotted configuration key; may be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and forward one batch per loader without creating run artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run stage-two training and return a process-compatible status code."""
    arguments = _argument_parser().parse_args(argv)
    config = load_config(arguments.config, arguments.set)
    validate_stage2_config(config)
    if arguments.dry_run:
        run_dry_run(config)
        return 0
    runtime = build_runtime(config)
    run_training(runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
