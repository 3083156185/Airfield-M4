# Adapting Single-Sensor Detectors to Multimodal Input

This repository releases D-FINE as the reference implementation. The same
input and fusion contracts can be applied to CNN-, DETR-, and Mamba-based
detectors without copying their third-party repositories into this project.

## 1. Common input contract

Each sample contains an RGB tensor and one spatially aligned auxiliary tensor:

```python
sample = {
    "rgb": rgb_tensor,  # [3, H, W]
    "npy": aux_tensor,  # [C_aux, H, W]
}
```

The auxiliary tensor may represent thermal infrared, depth, or another dense
sensor measurement. Its filename stem must match the RGB filename stem. Apply
the same crop, resize, flip, mosaic, and mixup geometry to every modality. A
photometric transform intended for RGB should not be applied to physical
measurements unless its meaning is explicitly defined for that sensor.

The reference implementation is located in:

- `engine/data/dataset/multimodal_coco_dataset.py`
- `engine/data/transforms/multimodal_container.py`
- `engine/data/dataloader.py`

## 2. Four fusion baselines

Assume an RGB stream with stage features `R_l` and an auxiliary stream with
features `A_l` at pyramid level `l`.

### Early fusion

Concatenate raw inputs and adapt the first convolution:

```python
x = torch.cat([rgb, aux], dim=1)
```

This is the smallest code change, but it forces heterogeneous measurements to
share a representation before modality-specific low-level patterns are formed.

### Plain middle fusion

Run two independent backbones and combine only aligned stage outputs:

```python
F_l = project(torch.cat([R_l, A_l], dim=1))
```

No fused feature is fed into the next RGB stage. This baseline isolates the
benefit of feature-level fusion from progressive cross-modal interaction.

### PAPF

Use RGB as the primary stream and progressively inject auxiliary features:

```python
R_next = rgb_stage(torch.cat([R_l, A_l], dim=1))
A_next = aux_stage(A_l)
```

Repeat the operation at P3, P4, and P5. The auxiliary branch remains compact,
while the primary stream accumulates complementary information across scales.
The D-FINE graph is defined in
`configs/multimodal/yaml/dfine_s_papf.yml`.

### Late-neck fusion

Keep both backbones and necks independent, then fuse final scale-aligned neck
features before the detection decoder. This preserves modality-specific
processing longest, at the cost of duplicating more computation.

## 3. Architecture-specific insertion points

### CNN one-stage detectors

1. Modify the dataloader to return the common sample dictionary.
2. Add a lightweight auxiliary stem and stage sequence.
3. For PAPF, inject the auxiliary feature before each downsampling stage of the
   primary backbone.
4. Project fused channels back to the original neck width so the prediction
   head remains unchanged.

### DETR-family detectors

1. Produce aligned P3/P4/P5 features for RGB and the auxiliary modality.
2. Apply the selected fusion policy before the multiscale transformer encoder.
3. Keep positional encoding, query initialization, decoder, matcher, and loss
   unchanged to preserve a controlled comparison.

### Mamba-based detectors

1. Add a compact auxiliary stem with the same spatial strides as the primary
   stream.
2. Fuse before the corresponding state-space stage, then project to the channel
   width expected by that stage.
3. Avoid flattening or scanning modalities independently after their spatial
   grids become inconsistent.

## 4. Fair-comparison checklist

- Use the same train/validation split, annotations, input resolution, epochs,
  optimizer, augmentation schedule, and random seed.
- Match RGB and auxiliary samples by image ID and verify spatial dimensions.
- Report RGB-only and auxiliary-only baselines before reporting fusion gains.
- Report parameter count and inference speed because late or symmetric
  dual-stream designs can add substantially more computation.
- Keep the detector head and loss unchanged when the experiment is intended to
  measure only the effect of the fusion policy.

## 5. Minimal implementation map

| Concern | D-FINE reference | Porting action |
|---|---|---|
| Paired data | `MultimodalCocoDetection` | Return aligned RGB and auxiliary tensors |
| Batch collation | `BatchMultimodalCollateFunction` | Preserve the modality dictionary |
| Early fusion | `BatchEarlyFusionCollateFunction` | Concatenate channels before the backbone |
| Tensor selection | `GetRGBModalityTensor`, `GetNPYModalityTensor` | Add equivalent graph nodes or forward branches |
| PAPF graph | `dfine_s_papf.yml` | Inject auxiliary features stage by stage |
| Plain middle | `dfine_s_plain_middle.yml` | Fuse only final backbone stages |
| Late neck | `dfine_s_late_neck.yml` | Fuse final neck features before the head |
