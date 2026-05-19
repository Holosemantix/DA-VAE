# 32x DA-VAE Architecture Design

## Goal

The current production VAE is a 16x KL autoencoder: an input image `x` is encoded to latent `z16` with spatial size `H/16 x W/16` and 32 channels. The 32x DA-VAE keeps the existing 16x encoder/decoder as the reconstruction backbone, then adds one DA-style deep-compression stage so the trainable latent becomes `H/32 x W/32`.

Recommended first target:

| Model | Latent shape for 1024x1024 | Token count | Channel |
| --- | --- | ---: | ---: |
| current f16c32 | `32 x 64 x 64` | 4096 | 32 |
| proposed f32c64 | `64 x 32 x 32` | 1024 | 64 |

`f32c64` gives 4x fewer spatial tokens while retaining half of the scalar latent capacity of f16c32. If reconstruction quality is insufficient, increase the 32x latent channels to 128 without changing the spatial architecture.

## Network

The model has a frozen 16x teacher path and a trainable 32x student path.

### Frozen Teacher Path

The teacher is the existing 16x VAE encoder, loaded from the current f16c32 checkpoint and frozen.

To align spatial sizes with the 32x student, the teacher encodes a half-resolution image:

```text
x:        B x 3  x H     x W
x_down:   B x 3  x H/2   x W/2
teacher:  B x 32 x H/32  x W/32
```

The teacher latent `z_b = mode(q_16(x_down))` is only used as semantic/alignment supervision. It is not decoded and it receives no gradient.

### Trainable Student Path

The student reuses the 16x Swin VAE encoder/decoder you provided:

```text
Encoder stem + ResBlocks + Swin bottleneck + 2x patchify
    -> 16x Gaussian moments, 2 * C16 channels at H/16 x W/16
```

Then a new deep-compression down block converts the 16x moments to 32x moments:

```text
moments16: B x (2*C16) x H/16 x W/16
DCDown2d:  conv + pixel_unshuffle shortcut, factor=2
moments32: B x (2*C32) x H/32 x W/32
posterior q_32(z|x) = DiagonalGaussian(moments32)
```

Default channels:

```text
C16 = 32
C32 = 64
moments16 channels = 64
moments32 channels = 128
```

The decoder mirrors this with a deep-compression up block:

```text
z32:      B x C32 x H/32 x W/32
DCUp2d:   conv + pixel_shuffle shortcut, factor=2
z16_hat:  B x C16 x H/16 x W/16
decoder:  existing 16x Swin VAE decoder
x_rec:    B x 3 x H x W
```

The Swin attention remains in the original 16x bottleneck. The added 32x compression is intentionally local and lightweight so existing 16x weights can initialize most of the model.

## Alignment Head

Because student channels differ from teacher channels, add one latent alignment head:

```text
z32:       B x C32 x H/32 x W/32
align(z):  B x 32  x H/32 x W/32
```

Two supported modes:

1. `mean`: group-average channels, e.g. `64 -> 32`; no extra parameters.
2. `proj`: learned `1x1 Conv2d(C32, 32)`.

For the first run, use `mean` because it is stable and matches the DA-VAE alignment variant already present in this repo.

## Loss

Use the same loss family as DA-VAE:

```text
L_total =
    L_rec
  + lambda_kl  * KL(q_32(z|x) || N(0, I))
  + lambda_gan * L_G
  + lambda_vf  * L_align
  + optional lambda_pe * L_patch_embed
```

Where:

`L_rec` is pixel reconstruction with perceptual LPIPS:

```text
L_rec = |x - x_rec| + lambda_lpips * LPIPS(x, x_rec)
```

`L_G` and discriminator loss follow the existing hinge GAN setup from `LPIPSWithDiscriminator`.

`L_align` is the DA-VAE semantic alignment between the frozen teacher latent and the mapped student latent:

```text
L_align = MSE(align(z32), z_b)
```

or, if using the original relation-preserving VF mode, the loss can compare cosine/distance matrices of `align(z32)` and `z_b`. The first 32x run should use MSE with `align_method=mean`, because teacher and student are already spatially matched.

Optional PatchEmbed alignment can be added later when adapting a DiT:

```text
L_patch_embed = MSE(PE_student(z32), PE_teacher(z_b))
```

This is useful for the second training stage with DiT, but it is not required for the first reconstruction/alignment stage.

## Training Stages

### Stage 0: Initialize From Current 16x VAE

Load the existing f16c32 checkpoint into both paths:

1. Teacher 16x VAE: frozen.
2. Student 16x encoder/decoder: trainable, initialized from the same checkpoint.
3. New `DCDown2d`, `DCUp2d`, and optional `align_proj`: randomly initialized.

### Stage 1: Reconstruction + Semantic Alignment

Train only the 32x VAE, without DiT/edit model training.

Use only GT images from the editing dataloader, normally `batch["edited_img"]`. Ignore low-quality images, masks, reference images, prompts, text encoders, Flux/Klein transformer, and flow-matching losses.

Recommended start:

```text
resolution: 1024 or multi-ratio GT crop
latent: f32c64
optimizer: AdamW
lr: 1e-4 for new DC blocks, or 5e-5/1e-4 for full student
kl_weight: 1e-6
disc_start: 5001
disc_weight: 0.1
vf_weight: 0.5
mixed_precision: bf16
```

Early diagnostics:

1. Reconstruction grid: GT vs reconstruction.
2. `train/rec_loss`, `train/kl_loss`, `train/vf_loss`, `train/disc_loss`.
3. Latent shape check: `B x 64 x H/32 x W/32`.
4. Teacher/student alignment: `align(z32)` and `z_b` must have identical shape.

### Stage 2: DiT Adaptation

After Stage 1 converges, replace the edit model's VAE with this 32x VAE and train only the DiT/token interface first. The DA-VAE paper's idea is to make the new compressed latent space semantically readable before jointly tuning the generator. For the current request, this stage is intentionally not implemented.

## Integration Boundary

From the edit training工程, keep:

- `MultiRatio` dataloader and GT image tensor.
- `Accelerator`/DeepSpeed distributed setup.
- visualization image saving and remote `mox.file.copy_parallel`.
- checkpoint save/resume.

Remove for this stage:

- Qwen tokenizer/text encoder.
- Flux/Klein transformer and scheduler.
- degraded/reference/mask conditioning.
- prompt dropout and CFG.
- flow matching noise/timestep loss.

