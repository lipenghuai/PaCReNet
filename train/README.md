# PaCReNet: Two-stage training and model details

See the [root README](../README.md) for installation, data preparation, and complete commands, and the [data format documentation](../docs/DATA_FORMAT.md) for input requirements. Run all commands from the repository root.

The default configurations use `datasets/pig_rgbd_dataset_split.json` and `runs/`. Real data and weights are not included. Stage 2 requires an explicit stage 1 checkpoint. Replace example paths with real, authorized inputs before running.

## Stage 1 connectivity constraints

The stage 1 configuration includes `connectivity.loss_weight: 0.1`, `connectivity.neighborhood: 8`, and `connectivity.postprocess: true`. The objective adds `loss_weight × L_conn` to the weighted cross-entropy or focal loss. It constrains background (0) and body (1, the full pig body), but not edge (2). Each class score is `sigmoid(class logit − maximum logit of the other classes)`; its 0.5 threshold matches the three-class argmax boundary. Zero-dimensional persistent homology is computed on each frame's 2D patch grid to strengthen the main component and penalize the persistence of extra components. Loss at critical locations is computed from the original torch tensors, allowing gradients to reach fragment peaks and connection breaks. The implementation draws on [Clough et al.'s topological loss](https://arxiv.org/abs/1910.01877) and applies only connected-component constraints.

The loss is applied separately for each frame and class. A class contributes a topology term only when its ground-truth patch labels have exactly one connected component. The term is skipped when the class is disconnected or absent, while the original classification supervision remains active. Skipping one class does not disable the constraint on the other. Connectivity loss is a soft constraint and requires retraining to affect network weights.

Postprocessing operates on discrete predictions. It first keeps the largest body component and relabels other body fragments as background, then keeps the largest background component and relabels other background fragments as edge. Ties retain the component encountered first in row-major order. If background was originally absent, discarded body fragments become edge instead. An absent background or body class is never created. Each of these classes therefore has at most one connected component when present. This does not guarantee connectivity of the combined body-plus-edge region or reconstruct occluded parts.

Training logs compute `background_single_component_rate`, `full_pig_single_component_rate`, and `both_single_component_rate` per frame from raw argmax predictions. An absent class fails the single-component condition; logs also record `background_empty_rate`, `full_pig_empty_rate`, and mean component counts. When postprocessing is enabled, evaluation saves a separate `postprocessed` report, with corresponding `postprocessed.*` CSV columns. Checkpoint selection still uses raw validation accuracy. In stage 2, `connectivity.postprocess` applies the same postprocessing before background handling and regional routing. Use the same setting during inference as during stage 2 training.

The stage 1 checkpoint structure is unchanged. To use the contributor alone, call `model.predict_classes(inputs, enforce_connectivity=True, connectivity=8)` after `model.eval()`. Calling `model(inputs)` still returns raw logits; applying `argmax` yourself does not perform correction. `TwoStagePigWeightModel(..., enforce_connectivity=True, connectivity=8)` explicitly enables correction. Older configurations without `connectivity` retain their previous loss and decoding behavior.

New stage 2 checkpoints also save `contribution_decoding`, which records the postprocessing flag and neighborhood actually used by the training model. Reconstruct the regression model with `TwoStagePigWeightModel(contributor, regressor, **payload.get("contribution_decoding", {"enforce_connectivity": False, "connectivity": 8}))`. Older checkpoints without this field retain decoding without postprocessing.

Disable the topology term with `--set connectivity.loss_weight=0.0`, disable postprocessing with `--set connectivity.postprocess=false`, or use a 4-neighborhood with `--set connectivity.neighborhood=4`. Component identification runs on the CPU and adds training and inference overhead. Start with a weight of 0.1 and adjust using raw connectivity rates and classification accuracy.

## Input modalities

Stage 2 controls evaluation batch size separately through `training.evaluation_batch_size` and pinned host memory through `training.pin_memory`. For older configurations without these settings, evaluation defaults to `min(batch_size, 8)` and pinned memory is disabled on Windows. Stage 2 batches only the modalities used by the two models, along with identifiers and weights. When the models use the same modality, they reuse the transferred tensor. Gradient buffers are released after each training epoch, and unused CUDA cache is released between training and evaluation. The current configurations use batch size 4 for both. If memory is limited on Windows, reduce each to 1–2 and keep `pin_memory: false`. These settings reduce peak memory usage; when system committed memory approaches the combined RAM and page-file capacity, reduce other programs' memory usage or enlarge the page file.

| Value | Channels | Processing |
|---|---:|---|
| `rgb` | 3 | Convert to `[0,1]`, then standardize with ImageNet mean/std |
| `depth` | 1 | Clip to `[0, depth_max_mm]`, then divide by `depth_max_mm`; 0 remains invalid depth |
| `rgbd` | 4 | Concatenate standardized RGB and normalized depth |

Stage 1 uses `model.modality`. Stage 2 restores the contributor modality from the specified stage 1 checkpoint; `model.regressor_modality` independently selects the regression modality. RGB, DEPTH, DEPTH-SEG, and MASK must share the same native `480×640` grid, without resizing or cropping.

## Model profiles

| Profile | Contributor: dim / depth / heads | Regressor: dim / depth / heads |
|---|---|---|
| `tiny` | 96 / 2 / 3 | 192 / 12 / 3 |
| `small` | 192 / 4 / 3 | 384 / 12 / 6 |
| `base` | 384 / 6 / 6 | 768 / 12 / 12 |
| `large` | 512 / 8 / 8 | 1024 / 24 / 16 |

Both stages default to `small`. `overrides` / `regressor_overrides` can override `embed_dim`, `depth`, `heads`, and `mlp_ratio`, but `embed_dim` must be divisible by `heads`. Reducing patch size from the default 32 to 16 raises the token count to 1200 and substantially increases GPU memory and compute requirements with `base` or `large`.

## Stage 2 background zeroing

The default `model.background_mode: zero` preserves the full `15×20` patch grid. The frozen stage 1 contributor still reads the original complete input. Only the stage 2 regression branch zeros patches predicted as background (class `0`) across all normalized input channels, including RGB and depth, before patch embedding. This is numerical zero in the normalized tensor; it is not produced by painting raw RGB pixels black before standardization. Original samples and stage 1 inputs are unchanged.

Background positions bypass the body and edge experts. Their embedded tokens are zeroed again after positional embeddings are added, ensuring all-zero background positions at the shared ViT input. Grid positions and order are preserved, and every frame enters the ViT with `300+1` tokens, including CLS. Background tokens still participate in subsequent attention. If every patch is predicted as background, regression continues with the full zero-background grid, without forcing any foreground patch to remain. In this case, `selected_counts` and `fallback_count` are both `0`.

Older configurations without `model.background_mode` default to `drop`, which removes background tokens and retains the previous all-background fallback behavior. Restore this behavior explicitly with `--set model.background_mode=drop`; configure the expert setting independently to match the checkpoint. The resolved run configuration and checkpoint field `regressor_metadata.background_mode` record the actual mode, and dry-run JSON also reports `background_mode`. Reconstruct older checkpoints without this field using `drop`. Reconstruct checkpoints trained with zeroing using their saved `zero` mode.

## Stage 2 body and edge experts

The default `train/configs/stage2.yaml` enables `model.expert_routing: true`. The frozen stage 1 contributor supplies fixed routing: class `1` (body) goes to `body_expert`, and class `2` (edge) goes to `edge_expert`. When connected-component postprocessing is enabled, routing uses the corrected classes. The experts are independently parameterized, trainable residual MLPs that transform their respective patch embeddings before the shared ViT. In the default zeroing mode, their outputs and zero background tokens enter the same ViT and weight regression head in the complete grid. Routing adds no trainable gate, requires no extra labels, and does not require retraining stage 1. The experts, shared ViT, and regression head are all trained by the stage 2 weight loss.

Older configurations without `model.expert_routing` disable experts by default. To reproduce the earlier stage 2 architecture without experts using the current configuration, add `--set model.expert_routing=false --set model.background_mode=drop`. Reconstruct older stage 2 checkpoints without expert weights with experts disabled. Checkpoints trained with experts contain both sets of expert weights and require experts to be enabled during reconstruction. The run's `resolved_config.yaml` explicitly records the setting; `regressor_metadata.expert_routing` in the checkpoint records the actual model setting. Older checkpoints without this field use `false`. Dry-run JSON also reports `expert_routing` and the total number of trainable expert parameters, `trainable_expert_parameters`.

When loading a stage 2 checkpoint, reconstruct the regressor from its saved metadata, then load the weights strictly:

```python
metadata = payload["regressor_metadata"]
regressor = PigWeightViT(
    {"rgb": 3, "depth": 1, "rgbd": 4}[metadata["modality"]],
    tuple(metadata["image_size"]),
    metadata["patch_size"],
    metadata["profile"],
    overrides=metadata["overrides"],
    expert_routing=metadata.get("expert_routing", False),
    background_mode=metadata.get("background_mode", "drop"),
)
regressor.load_state_dict(payload["regressor_state"], strict=True)
```

## Data splits and augmentation boundaries

Use a de-identified pig-level split manifest and keep every frame of each pig in one split. Neither stage includes cross-scene data in training by default. The current API, logs, and prediction files call the `test` split `validation` and use it for model selection. Cross-scene data is used only for reporting.

Augmentation runs only in the stage 1 optimization DataLoader, with a default probability of 0.5. The source pig's `MASK×RGB` and `DEPTH-SEG` overwrite the target frame at the original coordinates, without translation, rotation, scaling, removal of the target pig, or edge feathering. The output identity, weight, and labels all belong to the source pig.

Training-metric evaluation, validation, cross-scene evaluation, and all stage 2 paths run without augmentation. The default background pool uses only the stage 1 training split. Adding other splits to this pool risks leakage through scene pixels.

The default `data.invalid_sample_policy: skip` skips invalid files and records the reasons in the audit. Use `--set data.invalid_sample_policy=error` to stop on invalid samples. Training reads input images and split manifests without modifying them.

## Training metrics and run artifacts

Each epoch evaluates train, validation, and cross_scene_test without augmentation. Stage 1 selects the best checkpoint by validation patch accuracy and reports loss, accuracy, macro-F1, per-class metrics, and confusion matrices. Stage 2 selects by validation pig-level MAE and reports frame-level and pig-level MAE, RMSE, R², and MAPE. Cross-scene metrics are reported but do not affect selection.

Training creates a timestamped directory under `training.runs_root`:

```text
YYYYMMDD-HHMMSS_stage{1|2}_{modality}_{profile}\
├── resolved_config.yaml
├── environment.json
├── data_audit.json
├── split_manifest.json
├── metrics.jsonl
├── metrics.csv
├── checkpoints\best.pt
├── checkpoints\last.pt
└── predictions\
```

Stage 1 saves confusion matrices and classification summaries for each split in `predictions`. Stage 2 saves frame-level predictions and pig-level summary CSVs for each split. Stage 2 also records the normalized stage 1 checkpoint path and its SHA-256 in the configuration and environment metadata.

The project's smoke tests and dry runs do not perform full 50/100-epoch training. Only training commands without `--dry-run` begin optimization and create run artifacts.

Stage 2 checkpoints save only regressor weights and do not include stage 1 weights. Keep both checkpoints for deployment or reconstruction, and restore the saved decoding and architecture metadata. Run directories contain paths and sample identifiers; keep them private and review their contents before sharing results.
