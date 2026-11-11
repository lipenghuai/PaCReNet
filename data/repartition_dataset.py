"""Repartition non-cross-scene pigs without changing frames or labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _summary(pigs: list[dict]) -> dict:
    weights = np.array([pig["weight_kg"] for pig in pigs], dtype=float)
    if not len(weights):
        raise ValueError("training and same-scene test must both contain pigs")
    quantiles = np.quantile(weights, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "pig_count": len(pigs), "frame_count": sum(len(pig["frames"]) for pig in pigs),
        "weight_kg": {"min": float(weights.min()), "max": float(weights.max()),
            "mean": float(weights.mean()), "std_population": float(weights.std()),
            **dict(zip(("q10", "q25", "median", "q75", "q90"), quantiles.tolist()))},
        "scene_counts": dict(sorted(Counter(pig["scene"] for pig in pigs).items())),
    }


def repartition_document(document: dict, *, test_fraction: float = 0.2,
                         seed: int = 20261012) -> dict:
    """Allocate tests proportionally by scene, then sample weight-quantile blocks."""
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, (int, float)) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be a finite number between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    source = document["splits"]
    if set(source) not in ({"train", "validation", "cross_scene_test"},
                           {"train", "test", "cross_scene_test"}):
        raise ValueError("split keys must contain train, test (or validation), cross_scene_test")
    held_out = "test" if "test" in source else "validation"
    pool = source["train"] + source[held_out]
    all_pigs = pool + source["cross_scene_test"]
    ids = [pig["pig_id"] for pig in all_pigs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pig in input splits")
    cross_scenes = {pig["scene"] for pig in source["cross_scene_test"]}
    if any(pig["scene"] in cross_scenes for pig in pool):
        raise ValueError("cross-scene pigs must not appear in the repartition pool")
    if any(not math.isfinite(float(pig["weight_kg"])) or float(pig["weight_kg"]) <= 0 for pig in all_pigs):
        raise ValueError("pig weights must be finite and positive")

    scenes = defaultdict(list)
    for pig in pool:
        scenes[pig["scene"]].append(pig)
    test_count = math.floor(len(pool) * test_fraction + 0.5)
    capacity = sum(max(0, len(pigs) - 1) for pigs in scenes.values())
    if not 1 <= test_count <= capacity:
        raise ValueError("test_fraction leaves too few pigs to keep each scene in training")
    eligible = sum(len(pigs) > 1 for pigs in scenes.values())
    ideals = {scene: len(pigs) * test_count / len(pool) for scene, pigs in scenes.items()}
    minimums = {scene: int(len(pigs) > 1 and test_count >= eligible) for scene, pigs in scenes.items()}
    quotas = {scene: min(len(pigs) - 1, max(minimums[scene], math.floor(ideals[scene])))
              for scene, pigs in scenes.items()}
    while sum(quotas.values()) < test_count:
        scene = max((s for s in scenes if quotas[s] < len(scenes[s]) - 1),
                    key=lambda s: (ideals[s] - quotas[s], s))
        quotas[scene] += 1
    while sum(quotas.values()) > test_count:
        scene = max((s for s in scenes if quotas[s] > minimums[s]),
                    key=lambda s: (quotas[s] - ideals[s], s))
        quotas[scene] -= 1
    rng = random.Random(seed)
    test_ids = set()
    for scene in sorted(scenes):
        pigs = sorted(scenes[scene], key=lambda pig: (pig["weight_kg"], pig["pig_id"]))
        quota = quotas[scene]
        for index in range(quota):
            block = pigs[index * len(pigs) // quota:(index + 1) * len(pigs) // quota]
            test_ids.add(rng.choice(block)["pig_id"])
    ordered = sorted(pool, key=lambda pig: (pig["scene"], pig["pig_id"]))
    updated = copy.deepcopy(document)
    updated["splits"] = {
        "train": [copy.deepcopy(pig) for pig in ordered if pig["pig_id"] not in test_ids],
        "test": [copy.deepcopy(pig) for pig in ordered if pig["pig_id"] in test_ids],
        "cross_scene_test": copy.deepcopy(source["cross_scene_test"]),
    }
    train_summary = _summary(updated["splits"]["train"])
    test_summary = _summary(updated["splits"]["test"])
    cross_summary = copy.deepcopy(document.get("summary", {}).get("splits", {}).get("cross_scene_test"))
    if cross_summary is None:
        cross_summary = _summary(updated["splits"]["cross_scene_test"])
    updated["summary"] = {"total_pigs": len(all_pigs),
        "total_frames": sum(len(pig["frames"]) for pig in all_pigs),
        "splits": {"train": train_summary, "test": test_summary, "cross_scene_test": cross_summary}}
    train_weights = np.array([pig["weight_kg"] for pig in updated["splits"]["train"]], dtype=float)
    test_weights = np.array([pig["weight_kg"] for pig in updated["splits"]["test"]], dtype=float)
    support = np.sort(np.unique(np.concatenate((train_weights, test_weights))))
    ks = np.max(np.abs(np.searchsorted(np.sort(train_weights), support, side="right") / len(train_weights)
                       - np.searchsorted(np.sort(test_weights), support, side="right") / len(test_weights)))
    updated["description"] = "Pig-level training and same-scene test repartition; cross-scene test set preserved unchanged"
    updated["updated_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    updated["previous_refresh_changes"] = copy.deepcopy(document.get("refresh_changes", {}))
    updated["refresh_changes"] = {"split_assignment_changed": True,
        "cross_scene_test_changed": False, "pig_payloads_changed": False,
        "frames_changed": False, "weights_changed": False}
    updated["split_rules"] = {"unit": "pig", "same_scene_test_fraction": test_fraction,
        "same_scene_test_count": test_count, "seed": seed,
        "same_scene_test_policy": "Allocate test pigs proportionally by scene, then sample fixed-seed equal-frequency weight bins within each scene",
        "cross_scene_test_policy": "Preserve original cross-scene pigs, labels, frame lists, ordering, and image paths",
        "cross_scene_test_scenes": sorted(cross_scenes),
        "remaining_policy": "Use all remaining non-cross-scene pigs for training"}
    updated["distribution_comparison"] = {"test_vs_train": {"ks_statistic": float(ks),
        "mean_difference_kg": float(abs(test_weights.mean() - train_weights.mean())),
        "std_difference_kg": float(abs(test_weights.std() - train_weights.std()))}}
    updated.setdefault("quality_checks", {}).update({"original_split_assignments_preserved": False,
        "cross_scene_test_unchanged": True, "all_pig_payloads_preserved": True, "duplicate_pig_ids": 0})
    identifiers = updated.setdefault("identifiers", {})
    identifiers["previous_dataset_sha256"] = identifiers.get("dataset_sha256")
    identifiers["dataset_sha256"] = _digest({"dataset_root": document.get("dataset_root"),
        "pigs": sorted(all_pigs, key=lambda pig: pig["pig_id"])})
    identifiers["dataset_sha256_definition"] = "SHA256 of canonical dataset_root and pig payloads sorted by pig_id; excludes split assignment"
    identifiers["previous_split_assignment_sha256"] = identifiers.get("split_assignment_sha256")
    identifiers["split_assignment_sha256"] = _digest({s: sorted(pig["pig_id"] for pig in pigs)
                                                     for s, pigs in updated["splits"].items()})
    identifiers["cross_scene_payload_sha256"] = _digest(source["cross_scene_test"])
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("write to a separate output first; retain the original manifest as a backup")
    raw = args.source.read_bytes()
    source = json.loads(raw.decode("utf-8-sig"))
    updated = repartition_document(source, test_fraction=args.test_fraction, seed=args.seed)
    updated["identifiers"]["source_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "summary": updated["summary"],
                      "distribution": updated["distribution_comparison"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
