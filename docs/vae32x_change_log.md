# 32x DA-VAE 修改说明书

## 修改目标

本次修改的目标是把现有 16x VAE 迁移到 32x VAE，并先完成 DA-VAE 思路中的第一阶段：

```text
重建训练 + 语义对齐训练
```

本阶段不训练编辑模型，不接入 Qwen/text encoder，不训练 Flux/Klein Transformer，也不做 DiT 联合训练。当前阶段只使用编辑训练工程 dataloader 中的 GT 图像。

当前开发分支：

```text
da-vae-32x-pretrain
```

## 重要设计调整：32x channel 改为 128

上一版草案里默认使用 `f32c64`。这个设计可以跑，但从信息容量角度看不是最合理的主方案。

原因是：

```text
16x -> 32x 后，latent 空间 token 数减少 4 倍
```

如果 channel 只从 32 变成 64，总 latent 标量容量会变成原来的一半。为了在空间降采样 4 倍后尽量保持 latent 总容量，channel 从 32 变成 128 更合理：

```text
f16c32:  32  x H/16 x W/16
f32c128: 128 x H/32 x W/32
```

对于 1024x1024 图像：

| 模型 | latent shape | token 数 | channel | latent 标量总量 |
| --- | --- | ---: | ---: | ---: |
| f16c32 | `32 x 64 x 64` | 4096 | 32 | 131072 |
| f32c128 | `128 x 32 x 32` | 1024 | 128 | 131072 |

所以本次已经把默认配置和代码默认值从 `64` 调整为 `128`。

## 新增文件一：`docs/vae32x_architecture.md`

这是 32x VAE 的中文网络架构设计文档，主要写清楚：

- 为什么 32x VAE 首选 `f32c128`。
- 16x 到 32x 后 channel 变为 4 倍的原因。
- teacher path 和 student path 的整体结构。
- 32x student 如何复用现有 16x Swin VAE。
- `DCDown2d` 如何把 16x moments 压缩成 32x moments。
- `DCUp2d` 如何把 32x latent 还原给原 16x decoder。
- alignment head 为什么默认使用 `mean`。
- 损失函数如何保持和 DA-VAE 一致。
- Stage 0、Stage 1、Stage 2 的训练思路。
- 当前阶段保留和移除的编辑训练工程模块。

## 新增文件二：`lightningdit/tokenizer/vae32x_da.py`

这是 32x DA-VAE 的网络实现文件。

### `DCDownBlock2d`

功能：

```text
16x Gaussian moments -> 32x Gaussian moments
```

默认输入输出：

```text
输入 moments16: B x 64  x H/16 x W/16
输出 moments32: B x 256 x H/32 x W/32
```

其中：

```text
64  = 2 * C16  = 2 * 32
256 = 2 * C32  = 2 * 128
```

内部结构：

```text
主分支:   Conv2d -> pixel_unshuffle(factor=2)
shortcut: pixel_unshuffle(factor=2) -> channel group/proj
输出:     主分支 + shortcut
```

### `DCUpBlock2d`

功能：

```text
32x latent -> 原 16x decoder latent
```

默认输入输出：

```text
输入 z32:     B x 128 x H/32 x W/32
输出 z16_hat: B x 32  x H/16 x W/16
```

内部结构：

```text
主分支:   Conv2d -> pixel_shuffle(factor=2)
shortcut: channel repeat -> pixel_shuffle(factor=2)
输出:     主分支 + shortcut
```

### `DAVAE32xFrom16x`

这是核心模型类。

它包含：

- `student`
  - 可训练。
  - 复用现有 16x VAE encoder/decoder。
  - 新增 `extra_down` 和 `extra_up`。

- `teacher`
  - 冻结。
  - 加载现有 16x VAE checkpoint。
  - 输入半分辨率 GT 图像，输出和 student 32x latent 空间尺寸一致的 teacher latent。

关键接口：

```python
encode_student(x)
encode_teacher(x)
decode(z32)
forward(x)
```

`forward(x)` 返回：

```python
recon, student_posterior, extra
```

其中 `extra` 包含：

```python
extra["z_teacher"]        # B x 32  x H/32 x W/32
extra["z_student"]        # B x 128 x H/32 x W/32
extra["z_student_align"]  # B x 32  x H/32 x W/32
```

这些变量用于 DA-VAE 风格的语义对齐 loss。

## 新增文件三：`tools/train_vae32x_from_edit_gt.py`

这是 Stage 1 训练脚本，只训练 32x VAE。

保留的训练工程能力：

- `Accelerator` 分布式训练入口。
- 可继续使用 accelerate/deepspeed 配置。
- `MultiRatio` dataloader。
- checkpoint 保存。
- checkpoint resume。
- 训练集可视化。
- 验证集可视化。
- TensorBoard tracker。
- `mox.file.copy_parallel` 远端同步。

明确删除或绕开的编辑模型逻辑：

- Qwen tokenizer。
- Qwen/text encoder。
- Flux/Klein Transformer。
- diffusion scheduler。
- timestep/noise/flow matching。
- degraded image。
- reference image。
- mask。
- prompt/CFG dropout。

训练中实际使用的数据 key：

```python
batch["edited_img"]
```

如果你的 GT key 不叫 `edited_img`，可以在配置里改：

```yaml
image_key: "edited_img"
```

训练 loss 调用方式：

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

变量含义：

- `pixel_values`：GT 图像。
- `recon`：32x VAE 重建图像。
- `posterior`：32x KL posterior。
- `z_student_align`：student latent 映射到 teacher channel 后的结果。
- `z_teacher`：冻结 teacher latent。
- `z_student`：原始 32x student latent。

## 新增文件四：`configs/vae32x/train_vae32x_from_edit_gt.yaml`

这是 32x VAE Stage 1 训练的示例配置。

已经调整为：

```yaml
latent_channels_16x: 32
latent_channels_32x: 128
teacher_latent_channels: 32
align_method: "mean"
```

需要你根据真实训练环境修改的字段：

```yaml
dataset_config_name: "/path/to/train_dataset.yaml"
val_dataset_config_name: "/path/to/val_dataset.yaml"
student_ckpt_path: "/path/to/current_f16c32_vae.pth"
teacher_ckpt_path: "/path/to/current_f16c32_vae.pth"
target: f16c32_vae_swin_dinov3_dec.autoencoder_2d_16x_5layer_ResBlockList_swin_2d.AutoencoderKL
```

其中 `target` 必须保证在真实训练环境中可以 import。

## 没有修改的已有文件

本次没有改已有 DA-VAE 核心实现：

```text
lightningdit/davae/ldm/models/da_autoencoder.py
lightningdit/davae/ldm/modules/diffusionmodules/da_model.py
lightningdit/davae/ldm/modules/losses/contperceptual.py
```

本次也没有改已有编辑训练脚本，而是新增了一个专门的 Stage 1 VAE 训练脚本，避免影响现有编辑模型训练流程。

## 启动方式

真实训练：

```bash
accelerate launch tools/train_vae32x_from_edit_gt.py \
  --yml_path configs/vae32x/train_vae32x_from_edit_gt.yaml
```

最小 dummy dataset 检查：

```bash
accelerate launch tools/train_vae32x_from_edit_gt.py \
  --yml_path configs/vae32x/train_vae32x_from_edit_gt.yaml \
  --dummy_dataset
```

## 已完成验证

已通过语法检查：

```bash
python3 -m py_compile lightningdit/tokenizer/vae32x_da.py tools/train_vae32x_from_edit_gt.py
```

## 尚未完成的验证

当前本地默认 `python3` 环境缺少 `torch`，所以还没有完成：

- tensor shape smoke test。
- dummy dataset 单 step 训练。
- 真实 `MultiRatio` dataloader 训练。
- 真实 16x VAE checkpoint 加载检查。

这些需要在实际训练环境中完成。

## 后续建议

1. 先在训练环境跑 dummy dataset，确认模型能实例化、前反向能跑通。
2. 打印并确认 `z_student` shape 是 `B x 128 x H/32 x W/32`。
3. 打印并确认 `z_student_align` 和 `z_teacher` shape 都是 `B x 32 x H/32 x W/32`。
4. 先用 `align_method: mean` 得到稳定 baseline。
5. 如果后续 DiT 输入通道压力太大，可以再做 `f32c64` ablation。
6. Stage 1 收敛后，再进入 DiT/编辑模型适配阶段。

