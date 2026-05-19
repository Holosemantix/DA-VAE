# 32x DA-VAE 修改说明书

## 修改目标

本次修改建立了一个从现有 16x VAE 迁移到 32x VAE 的 Stage 1 训练方案。目标是先完成 DA-VAE 思路中的「重建 + 语义对齐」阶段，不引入编辑模型、文本编码器、参考图条件或 DiT/Transformer 联合训练。

当前分支：

```text
da-vae-32x-pretrain
```

## 新增文件

### `docs/vae32x_architecture.md`

这是 32x VAE 的网络架构设计文档，说明了：

- 为什么把现有 `f16c32` 改成第一版 `f32c64`。
- 32x student path 如何复用现有 16x Swin VAE encoder/decoder。
- 如何在 16x Gaussian moments 后额外增加一个 `2x` deep-compress block。
- 冻结 16x teacher path，并对半分辨率 GT 图像编码，生成空间尺寸一致的语义 latent。
- alignment head 的两种模式：`mean` 和 `proj`。
- 损失函数设计：重建/LPIPS/KL/GAN/VF alignment，和 DA-VAE 保持一致。
- Stage 0、Stage 1、Stage 2 的训练思路和当前阶段边界。

### `lightningdit/tokenizer/vae32x_da.py`

这是新增的 32x DA-VAE 网络实现，核心类为：

```python
DAVAE32xFrom16x
```

主要模块：

- `DCDownBlock2d`
  - 输入：现有 16x VAE encoder 输出的 Gaussian moments。
  - 功能：使用 `conv + pixel_unshuffle shortcut` 做额外 `2x` 空间压缩。
  - 默认把 `B x 64 x H/16 x W/16` 压缩为 `B x 128 x H/32 x W/32` moments。

- `DCUpBlock2d`
  - 输入：32x latent。
  - 功能：使用 `conv + pixel_shuffle shortcut` 恢复到原 16x decoder 需要的 latent 尺寸。
  - 默认把 `B x 64 x H/32 x W/32` 还原为 `B x 32 x H/16 x W/16`。

- `DAVAE32xFrom16x`
  - `student`：可训练 32x VAE path，复用现有 16x VAE encoder/decoder，并新增 `extra_down/extra_up`。
  - `teacher`：冻结 16x VAE path，用半分辨率 GT 图像编码得到 `H/32 x W/32` 的 teacher latent。
  - `encode_student()`：输出 32x KL posterior。
  - `encode_teacher()`：输出冻结 teacher posterior。
  - `decode()`：32x latent 先上采样回 16x latent，再走原 decoder。
  - `forward()`：返回 reconstruction、student posterior，以及 alignment 所需的 `z_teacher/z_student/z_student_align`。

默认 latent 设计：

```text
输入图像:      B x 3  x H     x W
student z32:  B x 64 x H/32  x W/32
teacher z:    B x 32 x H/32  x W/32
align(z32):   B x 32 x H/32  x W/32
```

### `tools/train_vae32x_from_edit_gt.py`

这是新增的 Stage 1 训练入口，只使用编辑训练工程 dataloader 里的 GT 图像。

保留的工程能力：

- `Accelerator` 分布式训练入口，因此可以继续使用 accelerate/deepspeed 配置。
- `MultiRatio` dataloader。
- checkpoint 保存与 resume。
- 训练/验证可视化。
- `mox.file.copy_parallel` 远端同步。
- TensorBoard tracker。

删除或绕开的编辑模型相关逻辑：

- Qwen tokenizer/text encoder。
- Flux/Klein transformer。
- scheduler、timestep、noise、flow matching。
- prompt、CFG dropout。
- degraded image、reference image、mask conditioning。

训练中实际使用的数据 key：

```python
batch["edited_img"]
```

如果训练数据的 GT key 不同，可以通过配置中的：

```yaml
image_key: "edited_img"
```

修改。

训练 loss 调用沿用 DA-VAE 的 `LPIPSWithDiscriminator`：

```python
loss_module(
    pixel_values,
    recon,
    posterior,
    optimizer_idx=0,
    z=extra["z_student_align"],
    aux_feature=extra["z_teacher"],
    z_pe=extra["z_student"],
    align_method="mean",
)
```

其中：

- `posterior` 用于 KL loss。
- `z_student_align` 和 `z_teacher` 用于 VF/semantic alignment。
- `recon` 和 `pixel_values` 用于 reconstruction/LPIPS/GAN。

### `configs/vae32x/train_vae32x_from_edit_gt.yaml`

这是新增的示例配置文件，包含：

- 训练参数。
- dataloader 配置路径占位。
- 16x VAE checkpoint 路径占位。
- 32x VAE 网络配置。
- DA-VAE loss 配置。

需要按实际环境修改的字段：

```yaml
dataset_config_name: "/path/to/train_dataset.yaml"
val_dataset_config_name: "/path/to/val_dataset.yaml"
student_ckpt_path: "/path/to/current_f16c32_vae.pth"
teacher_ckpt_path: "/path/to/current_f16c32_vae.pth"
target: f16c32_vae_swin_dinov3_dec.autoencoder_2d_16x_5layer_ResBlockList_swin_2d.AutoencoderKL
```

其中 `target` 需要保证在实际训练环境里可 import。

## 未修改的部分

本次没有修改已有 DA-VAE 源实现：

- `lightningdit/davae/ldm/models/da_autoencoder.py`
- `lightningdit/davae/ldm/modules/diffusionmodules/da_model.py`
- `lightningdit/davae/ldm/modules/losses/contperceptual.py`

本次也没有修改已有编辑训练脚本，而是新增了专门的 Stage 1 VAE 训练脚本，避免影响现有编辑模型训练流程。

## 训练方式

示例启动命令：

```bash
accelerate launch tools/train_vae32x_from_edit_gt.py \
  --yml_path configs/vae32x/train_vae32x_from_edit_gt.yaml
```

如果要先做最小 smoke test，可以启用 dummy dataset：

```bash
accelerate launch tools/train_vae32x_from_edit_gt.py \
  --yml_path configs/vae32x/train_vae32x_from_edit_gt.yaml \
  --dummy_dataset
```

注意：当前本地默认 `python3` 环境没有安装 `torch`，所以这里只完成了语法检查，真实 shape 和训练检查需要在训练环境中运行。

## 已完成验证

已通过语法检查：

```bash
python3 -m py_compile lightningdit/tokenizer/vae32x_da.py tools/train_vae32x_from_edit_gt.py
```

未完成验证：

- 未做 torch tensor shape smoke test，因为当前默认 Python 环境缺少 `torch`。
- 未做真实 dataloader 训练 step，因为当前仓库环境没有你训练环境中的 `common.flame` 数据依赖和 16x VAE 外部 package。

## 后续建议

1. 在真实训练环境里先跑 `--dummy_dataset` 或小 batch 检查模型实例化和 latent shape。
2. 确认 `student_ckpt_path/teacher_ckpt_path` 的 checkpoint key 是否能直接加载；如果 checkpoint 只保存裸 `state_dict` 或包了一层 `model/state_dict/ema`，当前代码都做了兼容。
3. 先用 `align_method: mean` 训练稳定 baseline。
4. 若重建细节不够，再把 `latent_channels_32x` 从 64 提到 128。
5. Stage 1 收敛后，再进入 DiT/编辑模型适配阶段。

