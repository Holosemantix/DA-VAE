# 32x DA-VAE 网络架构设计

## 设计目标

现有生产 VAE 是 16x KL AutoEncoder：输入图像 `x` 会被编码成空间尺寸为 `H/16 x W/16`、通道数为 32 的 latent。现在要把降采样倍率从 16x 提升到 32x，并尽量沿用 DA-VAE 的思想：

1. 先训练 VAE 的重建能力和语义对齐能力。
2. 再考虑和 DiT/编辑模型一起适配训练。

本阶段只做第 1 步，不引入编辑模型训练。

## 为什么 32x latent channel 建议变成 4 倍

从 16x 改成 32x 后，latent 的空间尺寸会从：

```text
H/16 x W/16
```

变成：

```text
H/32 x W/32
```

空间 token 数量减少 4 倍。若 channel 仍然保持 32，则 latent 总标量容量也会减少到原来的 1/4，重建细节压力会非常大。

因此更合理的第一版设计是把通道数从 32 提升到 128：

| 模型 | 1024x1024 图像的 latent shape | 空间 token 数 | latent channel | latent 标量总量 |
| --- | --- | ---: | ---: | ---: |
| 当前 f16c32 | `32 x 64 x 64` | 4096 | 32 | 131072 |
| 建议 f32c128 | `128 x 32 x 32` | 1024 | 128 | 131072 |

这样 32x VAE 在空间 token 数减少 4 倍的同时，总 latent 标量容量和现有 16x VAE 基本保持一致，更符合 DA-VAE「把空间信息压到通道维」的设计直觉。

如果后续为了降低 DiT 输入通道或显存，也可以试 `f32c64`，但它会把总 latent 容量降到原 16x 的一半，更适合作为轻量 ablation，而不是首选主方案。

## 总体网络结构

32x DA-VAE 由两条路径组成：

1. 冻结的 16x teacher path：提供语义对齐目标。
2. 可训练的 32x student path：负责重建和最终编码/解码。

整体结构：

```text
输入图像 x
  ├─ teacher path: resize 到 1/2 分辨率 -> 冻结 16x VAE encoder -> z_teacher
  └─ student path: 原 16x VAE encoder 的 conv_out 前高维特征 -> 额外 2x deep-compress -> z32

z32 -> alignment head -> z_student_align
z_student_align 与 z_teacher 做语义对齐

z32 -> 额外 2x deep-uncompress -> 原 16x VAE decoder 的 conv_in 后接入点 -> 重建图像 x_rec
```

## 冻结 Teacher Path

teacher 使用当前已有的 16x VAE encoder，并加载现有 f16c32 checkpoint。teacher 完全冻结，只用于产生语义对齐监督。

为了让 teacher latent 和 student 32x latent 的空间尺寸一致，teacher 输入不是原图，而是半分辨率图像：

```text
x:         B x 3  x H     x W
x_down:    B x 3  x H/2   x W/2
teacher:   B x 32 x H/32  x W/32
```

teacher latent 定义为：

```text
z_teacher = mode(q_16(x_down))
```

它只作为 alignment target，不参与解码，也不反传梯度。

## 可训练 Student Path

student 复用你给的 16x Swin VAE encoder/decoder。原始 16x encoder 的核心结构保持不变：

```text
Conv stem
  -> ResBlock down stages
  -> Swin bottleneck
  -> 2x patchify
  -> 16x preconv feature
  -> conv_out
  -> 16x Gaussian moments
```

DA-VAE 的关键是不要在已经被 `conv_out` 压成 64 个 moments 通道之后再压缩，而是在 `conv_out` 之前的高维特征处插入 `DCDown2d`。对于你给的 16x Swin VAE，这个截断点是：

```text
preconv16: B x 2048 x H/16 x W/16
```

这里的 2048 来自原 encoder bottleneck 的 512 通道经过 `2x patchify` 后变成 `512 * 2 * 2`。这个位置的信息量明显高于最终 `conv_out` 后的 64 moments 通道，更符合 DA-VAE 的构建方式。

为了得到 32x latent，在 `preconv16` 后新增一个 DA-style deep-compress block：

```text
preconv16: B x 2048 x H/16 x W/16
DCDown2d:  conv + pixel_unshuffle shortcut, factor=2
moments32: B x 256 x H/32 x W/32
posterior q_32(z|x) = DiagonalGaussian(moments32)
z32:       B x 128 x H/32 x W/32
```

默认通道配置：

```text
preconv_channels = 2048
C32 = 128
moments32 channels = 2 * 128 = 256
```

## 32x Decoder

decoder 做上述过程的镜像。先把 32x latent 还原到原 16x decoder 的高维 preconv 特征，再跳过原 decoder 的 `conv_in`，从 mid/up blocks 开始解码：

```text
z32:      B x 128 x H/32 x W/32
DCUp2d:   conv + pixel_shuffle shortcut, factor=2
preconv16_hat: B x 2048 x H/16 x W/16
unpatchify:    B x 512  x H/8  x W/8
decoder:       原 16x Swin VAE decoder 的 mid/up/end
x_rec:    B x 3   x H    x W
```

这样做的好处是：

1. 大部分已有 16x VAE 权重可以直接复用。
2. 新增参数集中在 `DCDown2d/DCUp2d/alignment head`，调试边界清楚。
3. 32x latent 的空间 token 数减少 4 倍，后续接 DiT 时 token 压力显著降低。
4. 压缩发生在 `conv_out` 前的信息富集位置，而不是在最终 moments 后继续压缩。

## Alignment Head

student latent 的 channel 是 128，teacher latent 的 channel 是 32，因此需要一个 alignment head：

```text
z32:             B x 128 x H/32 x W/32
z_student_align: B x 32  x H/32 x W/32
z_teacher:       B x 32  x H/32 x W/32
```

支持两种方式：

### `mean`

把 128 个 channel 分成 32 组，每组 4 个 channel，对组内取均值：

```text
128 -> 32
```

优点：

- 无额外参数。
- 和 `32 -> 128` 的 4 倍通道扩展天然匹配。
- 第一阶段训练更稳定。

第一版默认使用 `mean`。

### `proj`

使用可学习的 `1x1 Conv2d(128, 32)`：

```text
z_student_align = Conv1x1(z32)
```

优点是表达能力更强，缺点是多了一个可学习映射，早期训练可能比 `mean` 更不稳定。建议在 `mean` baseline 稳定后再尝试。

## 损失函数

损失函数和 DA-VAE 保持一致：

```text
L_total =
    L_rec
  + lambda_kl  * L_kl
  + lambda_gan * L_gan
  + lambda_vf  * L_align
  + optional lambda_pe * L_patch_embed
```

### 重建损失

```text
L_rec = |x - x_rec| + lambda_lpips * LPIPS(x, x_rec)
```

这里沿用 DA-VAE 的 `LPIPSWithDiscriminator`，即像素重建和 perceptual loss 共同约束重建质量。

### KL 损失

student posterior 是 32x KL posterior：

```text
q_32(z|x) = DiagonalGaussian(moments32)
```

KL loss：

```text
L_kl = KL(q_32(z|x) || N(0, I))
```

建议初始权重：

```text
lambda_kl = 1e-6
```

### GAN 损失

继续使用 DA-VAE 里的 hinge GAN 逻辑：

```text
L_gan = generator adversarial loss
L_disc = discriminator hinge loss
```

建议：

```text
disc_start = 5001
disc_weight = 0.1
```

也可以先关闭 discriminator，只训练 reconstruction + alignment，确认主路径无问题后再打开。

### 语义对齐损失

第一版建议使用 MSE alignment：

```text
L_align = MSE(z_student_align, z_teacher)
```

其中：

```text
z_student_align: B x 32 x H/32 x W/32
z_teacher:       B x 32 x H/32 x W/32
```

这对应 DA-VAE 里的 VF/semantic alignment 思想：让新的高压缩 latent 在语义结构上接近原模型可理解的 latent 空间。

### 可选 PatchEmbed 对齐

后续进入 DiT 适配阶段时，可以增加 PatchEmbed 对齐：

```text
L_patch_embed = MSE(PE_student(z32), PE_teacher(z_teacher))
```

这一步主要用于让 DiT 的输入投影层更快适配 32x latent。当前 Stage 1 不是必须。

## 分阶段训练思路

### Stage 0：初始化

1. teacher 16x VAE 加载当前 f16c32 checkpoint，冻结。
2. student 16x encoder/decoder 加载同一个 f16c32 checkpoint，可训练。
3. 新增 `DCDown2d/DCUp2d` 随机初始化。
4. 默认 alignment 使用 `mean`，不新增 alignment 参数。

### Stage 1：只训练 32x VAE

目标：

```text
训练 32x VAE 的重建能力 + 语义对齐能力
```

只使用编辑训练 dataloader 里的 GT 图像，默认：

```python
batch["edited_img"]
```

明确不使用：

- degraded image。
- reference image。
- mask。
- prompt。
- text encoder。
- Flux/Klein transformer。
- flow matching noise/timestep loss。

推荐初始配置：

```text
latent: f32c128
optimizer: AdamW
lr: 1e-4
kl_weight: 1e-6
disc_start: 5001
disc_weight: 0.1
vf_weight: 0.5
mixed_precision: bf16
align_method: mean
```

需要重点检查：

1. reconstruction grid：GT vs reconstruction。
2. `train/rec_loss`。
3. `train/kl_loss`。
4. `train/vf_loss`。
5. `train/disc_loss`。
6. latent shape 是否为 `B x 128 x H/32 x W/32`。
7. `z_student_align` 和 `z_teacher` 是否都是 `B x 32 x H/32 x W/32`。

### Stage 2：和 DiT/编辑模型适配

Stage 1 收敛后，再把编辑模型里的 VAE 替换为这个 32x VAE。此时建议先冻结 VAE，只训练 DiT 的输入适配层或 LoRA，让编辑模型先理解新的 latent 空间。

这个阶段当前没有实现，避免把 VAE 重建阶段和编辑模型训练混在一起。

## 与编辑训练工程的集成边界

本阶段保留：

- `MultiRatio` dataloader。
- GT 图像数据。
- `Accelerator`/DeepSpeed 分布式训练能力。
- 可视化保存。
- checkpoint 保存与恢复。
- `mox.file.copy_parallel` 远端同步。

本阶段移除：

- Qwen tokenizer/text encoder。
- Flux/Klein transformer。
- scheduler/timestep/noise。
- reference/degraded/mask 条件。
- prompt dropout/CFG。
- flow matching loss。
