"""Command-line orchestration for stage-one patch contribution training."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.dataset import (
    DatasetIndex,
    FrameRecord,
    RGBDFrameDataset,
    audit_records,
    load_dataset_index,
)
from data.preprocessing import PreprocessConfig, make_patch_labels
from data.sampling import PigBalancedSampler, build_stage1_splits
from model.patch_contribution import PatchContributionNet
from model.connectivity import component_counts, decode_patch_classes
from model.two_stage_model import (
    STAGE1_CLASS_NAMES,
    STAGE1_SCHEMA_VERSION,
    Stage1Metadata,
    save_stage1_checkpoint,
)
from train.common import (
    build_grad_scaler,
    build_optimizer_scheduler,
    collect_environment,
    seed_everything,
    seed_worker,
    select_device,
)
from train.config import load_config, validate_stage1_config
from train.run_io import RunDirectory
from train.stage1_engine import build_patch_loss, evaluate_stage1, train_stage1_epoch
from train.topology_loss import topology_connectivity_loss


_INPUT_CHANNELS = {"rgb": 3, "depth": 1, "rgbd": 4}
_SPLIT_NAMES = ("train", "validation", "cross_scene_test")


@dataclass(slots=True)
class Stage1Runtime:
    """All mutable and immutable dependencies of a stage-one training run."""

    config: dict[str, Any]
    run: RunDirectory
    model: torch.nn.Module
    metadata: Stage1Metadata
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: Any | None
    class_weights: torch.Tensor
    train_dataset: RGBDFrameDataset
    train_sampler: PigBalancedSampler
    train_loader: Any
    evaluation_loaders: Mapping[str, Any]
    device: torch.device


def automatic_class_weights(counts: torch.Tensor | Sequence[int]) -> torch.Tensor:
    """Return inverse-square-root weights, mean-normalized before clamping."""
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.ndim != 1 or values.numel() != 3:
        raise ValueError("class counts must contain exactly three values")
    if not torch.all(torch.isfinite(values)) or torch.any(values < 0):
        raise ValueError("class counts must be finite and nonnegative")
    weights = torch.rsqrt(torch.clamp(values, min=1.0))
    weights = weights / weights.mean()
    return weights.clamp(0.25, 4.0).to(dtype=torch.float32)


def _count_patch_labels(
    records: Sequence[FrameRecord], preprocess: PreprocessConfig
) -> torch.Tensor:
    counts = torch.zeros(3, dtype=torch.int64)
    for record in records:
        with Image.open(record.mask_path) as image:
            mask = np.asarray(image) > 0
        labels = make_patch_labels(
            mask,
            preprocess.patch_size,
            preprocess.background_max,
            preprocess.full_min,
        )
        counts += torch.bincount(labels, minlength=3)
    if int(counts.sum().item()) == 0:
        raise ValueError("cannot compute class weights without training patches")
    return counts


def _resolve_class_weights(
    configured: Any, records: Sequence[FrameRecord], preprocess: PreprocessConfig
) -> tuple[torch.Tensor, list[int] | None]:
    if configured == "auto":
        counts = _count_patch_labels(records, preprocess)
        return automatic_class_weights(counts), counts.tolist()
    weights = torch.as_tensor(configured, dtype=torch.float32)
    if weights.ndim != 1 or weights.numel() != 3:
        raise ValueError("loss.class_weights must be 'auto' or contain three values")
    if not torch.all(torch.isfinite(weights)) or torch.any(weights < 0):
        raise ValueError("loss.class_weights must be finite and nonnegative")
    return weights, None


def _audit_index(
    index: DatasetIndex, invalid_sample_policy: str
) -> tuple[DatasetIndex, dict[str, Any]]:
    valid_by_split: dict[str, tuple[FrameRecord, ...]] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    for split_name in _SPLIT_NAMES:
        valid, report = audit_records(index.by_split[split_name], invalid_sample_policy)
        valid_by_split[split_name] = valid
        split_reports[split_name] = report
    return DatasetIndex(index.dataset_root, valid_by_split), {
        "dataset_root": str(index.dataset_root),
        "invalid_sample_policy": invalid_sample_policy,
        "input_count": sum(report["input_count"] for report in split_reports.values()),
        "valid_count": sum(report["valid_count"] for report in split_reports.values()),
        "skipped_count": sum(report["skipped_count"] for report in split_reports.values()),
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


def _preprocess_config(config: Mapping[str, Any]) -> PreprocessConfig:
    return PreprocessConfig(
        image_size=tuple(config["data"]["image_size"]),
        patch_size=config["data"]["patch_size"],
        depth_max_mm=config["data"]["depth_max_mm"],
        background_max=config["patch_labels"]["background_max"],
        full_min=config["patch_labels"]["full_min"],
    )


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
    background_splits = config["augmentation"]["background_splits"]
    if not isinstance(background_splits, list) or any(
        split not in _SPLIT_NAMES for split in background_splits
    ):
        raise ValueError(f"augmentation.background_splits must use {list(_SPLIT_NAMES)}")
    samples_per_epoch = config["data"]["samples_per_epoch"]
    if samples_per_epoch is not None:
        _require_positive_integer("data.samples_per_epoch", samples_per_epoch)


def _make_loader(
    dataset: RGBDFrameDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    sampler: PigBalancedSampler | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_runtime(config: dict[str, Any]) -> Stage1Runtime:
    """Audit data and construct a fully reproducible stage-one runtime."""
    validate_stage1_config(config)
    _validate_runtime_config(config)
    resolved_config = copy.deepcopy(config)
    training = resolved_config["training"]
    seed_everything(training["seed"])
    device = select_device(training["device"])

    raw_index = load_dataset_index(resolved_config["data"]["split_json"])
    valid_index, audit_report = _audit_index(
        raw_index, resolved_config["data"]["invalid_sample_policy"]
    )
    splits = build_stage1_splits(
        valid_index,
        resolved_config["data"]["stage1_cross_scene_fraction"],
        resolved_config["data"]["stage1_cross_scene_seed"],
    )
    if not splits.train_records:
        raise ValueError("stage-one training split has no valid records")

    preprocess = _preprocess_config(resolved_config)
    class_weights, class_counts = _resolve_class_weights(
        resolved_config["loss"]["class_weights"], splits.train_records, preprocess
    )
    resolved_config["loss"]["resolved_class_weights"] = class_weights.tolist()
    if class_counts is not None:
        resolved_config["loss"]["class_counts"] = class_counts

    background_split_names = resolved_config["augmentation"]["background_splits"]
    scene_pixel_leakage = any(name != "train" for name in background_split_names)
    if scene_pixel_leakage:
        resolved_config["scene_pixel_leakage_warning"] = True
        print(
            "WARNING: validation/cross-scene pixels are enabled as augmentation "
            "backgrounds; scene pixel leakage is possible."
        )

    run = RunDirectory.create(
        Path(training["runs_root"]),
        "stage1",
        resolved_config["model"]["modality"],
        resolved_config["model"]["profile"],
    )
    manifest_records = {
        "train": splits.train_records,
        "validation": splits.validation_records,
        "cross_scene_test": splits.cross_scene_records,
    }
    run.write_yaml("resolved_config.yaml", resolved_config)
    run.write_json("environment.json", collect_environment())
    run.write_json("data_audit.json", audit_report)
    run.write_json(
        "split_manifest.json",
        {
            "selected_cross_scene_pig_ids": list(splits.selected_cross_pig_ids),
            "splits": {
                split_name: _record_manifest(records)
                for split_name, records in manifest_records.items()
            },
        },
    )

    background_mapping = {
        "train": splits.train_records,
        "validation": valid_index.by_split["validation"],
        "cross_scene_test": valid_index.by_split["cross_scene_test"],
    }
    background_records = tuple(
        record
        for split_name in background_split_names
        for record in background_mapping[split_name]
    )
    train_dataset = RGBDFrameDataset(
        splits.train_records,
        preprocess_config=preprocess,
        background_records=background_records,
        augmentation_probability=resolved_config["augmentation"]["probability"],
        augmentation_seed=resolved_config["augmentation"]["seed"],
    )
    evaluation_datasets = {
        split_name: RGBDFrameDataset(records, preprocess_config=preprocess)
        for split_name, records in manifest_records.items()
    }
    samples_per_epoch = resolved_config["data"]["samples_per_epoch"]
    train_sampler = PigBalancedSampler(
        splits.train_records,
        num_samples=len(splits.train_records) if samples_per_epoch is None else samples_per_epoch,
        seed=training["seed"],
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
        device=device,
        seed=training["seed"],
        sampler=train_sampler,
    )
    evaluation_loaders = {
        split_name: _make_loader(
            dataset,
            batch_size=training["batch_size"],
            num_workers=training["num_workers"],
            device=device,
            seed=training["seed"] + index + 1,
        )
        for index, (split_name, dataset) in enumerate(evaluation_datasets.items())
    }

    image_size = tuple(resolved_config["data"]["image_size"])
    patch_size = resolved_config["data"]["patch_size"]
    patch_count = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    print(f"Stage 1 patch tokens per frame: {patch_count}")
    modality = resolved_config["model"]["modality"]
    model = PatchContributionNet(
        _INPUT_CHANNELS[modality],
        image_size,
        patch_size,
        resolved_config["model"]["profile"],
        overrides=resolved_config["model"]["overrides"],
    ).to(device)
    metadata = Stage1Metadata(
        schema_version=STAGE1_SCHEMA_VERSION,
        modality=modality,
        input_channels=_INPUT_CHANNELS[modality],
        image_size=image_size,
        patch_size=patch_size,
        class_names=STAGE1_CLASS_NAMES,
        depth_max_mm=preprocess.depth_max_mm,
        rgb_mean=preprocess.rgb_mean,
        rgb_std=preprocess.rgb_std,
        profile=resolved_config["model"]["profile"],
        overrides=resolved_config["model"]["overrides"],
    )
    optimizer, scheduler, _ = build_optimizer_scheduler(
        model.parameters(),
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        epochs=training["epochs"],
        steps_per_epoch=len(train_loader),
        warmup_epochs=training["warmup_epochs"],
        patch_count=patch_count,
        profiles=resolved_config["model"]["profile"],
    )
    scaler = build_grad_scaler(device, enabled=bool(training["amp"]))
    return Stage1Runtime(
        config=resolved_config,
        run=run,
        model=model,
        metadata=metadata,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        class_weights=class_weights,
        train_dataset=train_dataset,
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


def _dry_run_augmentation_metadata(batch: Mapping[str, Any]) -> dict[str, Any]:
    metadata = batch["augmentation_metadata"]
    return {
        "augmented": bool(metadata["augmented"][0].item()),
        "source_split": metadata["source_split"][0],
        "source_pig_id": metadata["source_pig_id"][0],
        "source_frame_id": metadata["source_frame_id"][0],
        "target_split": metadata["target_split"][0],
        "target_pig_id": metadata["target_pig_id"][0],
        "target_frame_id": metadata["target_frame_id"][0],
    }


def run_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    """Audit data and forward one batch per stage-one loader without writing."""
    validate_stage1_config(config)
    _validate_runtime_config(config)
    training = config["training"]
    seed_everything(training["seed"])
    device = select_device(training["device"])

    raw_index = load_dataset_index(config["data"]["split_json"])
    valid_index, audit_report = _audit_index(
        raw_index, config["data"]["invalid_sample_policy"]
    )
    splits = build_stage1_splits(
        valid_index,
        config["data"]["stage1_cross_scene_fraction"],
        config["data"]["stage1_cross_scene_seed"],
    )
    split_records = {
        "train": splits.train_records,
        "validation": splits.validation_records,
        "cross_scene_test": splits.cross_scene_records,
    }
    empty_splits = [name for name, records in split_records.items() if not records]
    if empty_splits:
        raise ValueError(
            "stage-one dry-run requires valid records in every split: "
            + ", ".join(empty_splits)
        )

    background_mapping = {
        "train": splits.train_records,
        "validation": valid_index.by_split["validation"],
        "cross_scene_test": valid_index.by_split["cross_scene_test"],
    }
    background_records = tuple(
        record
        for split_name in config["augmentation"]["background_splits"]
        for record in background_mapping[split_name]
    )
    source_record = next(
        (
            source
            for source in splits.train_records
            if any(target.pig_id != source.pig_id for target in background_records)
        ),
        None,
    )
    if source_record is None:
        raise ValueError(
            "stage-one dry-run cannot force augmentation without a different-pig target"
        )

    preprocess = _preprocess_config(config)
    optimization_dataset = RGBDFrameDataset(
        (source_record,),
        preprocess_config=preprocess,
        background_records=background_records,
        augmentation_probability=1.0,
        augmentation_seed=config["augmentation"]["seed"],
    )
    optimization_loader = _make_loader(
        optimization_dataset,
        batch_size=1,
        num_workers=0,
        device=device,
        seed=training["seed"],
    )
    evaluation_loaders = {
        split_name: _make_loader(
            RGBDFrameDataset((records[0],), preprocess_config=preprocess),
            batch_size=1,
            num_workers=0,
            device=device,
            seed=training["seed"] + index + 1,
        )
        for index, (split_name, records) in enumerate(split_records.items())
    }
    loaders = {"optimization_train": optimization_loader, **evaluation_loaders}

    image_size = tuple(config["data"]["image_size"])
    patch_size = config["data"]["patch_size"]
    token_count = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    modality = config["model"]["modality"]
    model = PatchContributionNet(
        _INPUT_CHANNELS[modality],
        image_size,
        patch_size,
        config["model"]["profile"],
        overrides=config["model"]["overrides"],
    ).to(device)
    model.eval()

    batches: dict[str, dict[str, Any]] = {}
    connectivity_settings = config.get("connectivity", {})
    neighborhood = connectivity_settings.get("neighborhood", 8)
    forced_augmentation: dict[str, Any] | None = None
    with torch.no_grad():
        for loader_name, loader in loaders.items():
            batch = next(iter(loader))
            inputs = batch[modality].to(device)
            logits = model(inputs)
            batches[loader_name] = {
                "input_shape": list(inputs.shape),
                "output_shape": list(logits.shape),
                "raw_components": component_counts(logits.argmax(-1), model.grid_size, neighborhood).tolist(),
            }
            if connectivity_settings.get("loss_weight", 0.0) > 0:
                batches[loader_name]["topology_loss"] = float(topology_connectivity_loss(
                    logits, batch["patch_labels"].to(device), model.grid_size, neighborhood
                ).item())
            if connectivity_settings.get("postprocess", False):
                decoded = decode_patch_classes(
                    logits, model.grid_size, enforce_connectivity=True, connectivity=neighborhood
                )
                batches[loader_name]["postprocessed_components"] = component_counts(
                    decoded, model.grid_size, neighborhood
                ).tolist()
            if loader_name == "optimization_train":
                forced_augmentation = _dry_run_augmentation_metadata(batch)
    if forced_augmentation is None or not forced_augmentation["augmented"]:
        raise RuntimeError("stage-one dry-run failed to force in-memory augmentation")

    report = {
        "stage": "stage1",
        "mode": "dry_run",
        "device": str(device),
        "image_size": list(image_size),
        "patch_size": patch_size,
        "token_count": token_count,
        "connectivity": dict(connectivity_settings),
        "split_counts": {
            name: _dry_run_split_counts(records)
            for name, records in split_records.items()
        },
        "audit": {
            "input_count": audit_report["input_count"],
            "valid_count": audit_report["valid_count"],
            "skipped_count": audit_report["skipped_count"],
        },
        "selected_cross_scene_pig_ids": list(splits.selected_cross_pig_ids),
        "forced_augmentation": forced_augmentation,
        "batches": batches,
    }
    print(json.dumps(report, sort_keys=True))
    return report


def _print_report(epoch: int, split: str, report: Mapping[str, Any]) -> None:
    recalls = [float(row["recall"]) for row in report["per_class"]]
    connectivity_text = (
        f" {float(report['background_single_component_rate']):8.5f}"
        f" {float(report['full_pig_single_component_rate']):9.5f}"
        if "background_single_component_rate" in report else "      n/a       n/a"
    )
    print(
        f"{epoch:5d} {split:16s} {float(report['loss']):10.5f} "
        f"{float(report['accuracy']):10.5f} {float(report['macro_f1']):10.5f} "
        f"{recalls[0]:10.5f} {recalls[1]:10.5f} {recalls[2]:10.5f}"
        + connectivity_text
    )


def run_training(runtime: Stage1Runtime) -> None:
    """Train, evaluate all splits every epoch, checkpoint, and export diagnostics."""
    epochs = _require_positive_integer("training.epochs", runtime.config["training"]["epochs"])
    connectivity_settings = runtime.config.get("connectivity", {})
    neighborhood = connectivity_settings.get("neighborhood", 8)
    postprocess = connectivity_settings.get("postprocess", False)
    loss_function = build_patch_loss(
        runtime.config["loss"]["name"],
        runtime.class_weights,
        runtime.config["loss"]["focal_gamma"],
        grid_size=getattr(runtime.model, "grid_size", None),
        connectivity_weight=connectivity_settings.get("loss_weight", 0.0),
        connectivity=neighborhood,
    )
    best_validation_accuracy = -math.inf
    print(
        "epoch split                  loss   accuracy   macro_f1 "
        "recall_bg recall_full recall_edge conn_bg conn_full"
    )
    for epoch_index in range(epochs):
        epoch_number = epoch_index + 1
        runtime.train_dataset.set_epoch(epoch_index)
        runtime.train_sampler.set_epoch(epoch_index)
        train_stage1_epoch(
            runtime.model,
            runtime.train_loader,
            runtime.optimizer,
            runtime.device,
            modality=runtime.config["model"]["modality"],
            class_weights=runtime.class_weights,
            scaler=runtime.scaler,
            grad_clip=runtime.config["training"]["grad_clip"],
            scheduler=runtime.scheduler,
            loss_function=loss_function,
            show_progress=True,
            progress_description=f"Stage 1 epoch {epoch_number}/{epochs}",
            connectivity=neighborhood,
        )

        evaluation_reports: dict[str, dict[str, Any]] = {}
        for split_name in _SPLIT_NAMES:
            report = evaluate_stage1(
                runtime.model,
                runtime.evaluation_loaders[split_name],
                runtime.device,
                runtime.config["model"]["modality"],
                runtime.class_weights,
                loss_function=loss_function,
                enforce_connectivity=postprocess,
                connectivity=neighborhood,
            )
            evaluation_reports[split_name] = report
            runtime.run.append_metrics(epoch_number, split_name, report)
            _print_report(epoch_number, split_name, report)

        validation_accuracy = float(evaluation_reports["validation"]["accuracy"])
        is_best = validation_accuracy > best_validation_accuracy
        if is_best:
            best_validation_accuracy = validation_accuracy
        checkpoint_arguments = {
            "model": runtime.model,
            "metadata": runtime.metadata,
            "epoch": epoch_number,
            "optimizer_state": runtime.optimizer.state_dict(),
            "scheduler_state": runtime.scheduler.state_dict(),
            "best_validation_accuracy": best_validation_accuracy,
        }
        save_stage1_checkpoint(
            runtime.run.checkpoint_path("last.pt"), **checkpoint_arguments
        )
        if is_best:
            save_stage1_checkpoint(
                runtime.run.checkpoint_path("best.pt"), **checkpoint_arguments
            )

    best_payload = torch.load(
        runtime.run.checkpoint_path("best.pt"),
        map_location=runtime.device,
        weights_only=False,
    )
    runtime.model.load_state_dict(best_payload["model_state"], strict=True)
    for split_name in _SPLIT_NAMES:
        report = evaluate_stage1(
            runtime.model,
            runtime.evaluation_loaders[split_name],
            runtime.device,
            runtime.config["model"]["modality"],
            runtime.class_weights,
            loss_function=loss_function,
            enforce_connectivity=postprocess,
            connectivity=neighborhood,
        )
        runtime.run.write_confusion_matrix(
            split_name, report["confusion_matrix"], STAGE1_CLASS_NAMES
        )
        runtime.run.write_patch_class_summary(
            split_name, report["per_class"], STAGE1_CLASS_NAMES
        )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "stage1.yaml",
        help="Path to the stage-one YAML configuration.",
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
    """Run stage-one training and return a process-compatible status code."""
    arguments = _argument_parser().parse_args(argv)
    config = load_config(arguments.config, arguments.set)
    validate_stage1_config(config)
    if arguments.dry_run:
        run_dry_run(config)
        return 0
    runtime = build_runtime(config)
    run_training(runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
