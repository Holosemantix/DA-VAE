# 32x DA-VAE 训练地图 (Training Map)

> 本文档基于 `da-vae-32x-pretrain` 分支最新代码，梳理从配置文件到训练循环的完整训练链路。

---

## 1. 入口与配置文件

### 1.1 训练入口脚本

```
tools/train_vae32x_from_edit_gt.py
```

这是唯一的训练入口。它使用 `accelerate` 做分布式封装，单卡/多卡/FSDP 都走这个文件。

### 1.2 配置文件

```
configs/vae32x/train_vae32x_from_edit_gt.yaml
```

配置分为三大块：

| 区块 | 作用 |
|------|------|
| `args` | 训练超参：学习率、batch size、scheduler、checkpoint 频率等 |
| `model` | 模型结构定义：`DAVAE32xFrom16x` 及其子模块配置 |
| `loss` | 损失函数定义：`LPIPSWithDiscriminator` 及其权重 |

关键配置速览：

```yaml
args:
  learning_rate: 1.0e-4
  discriminator_learning_rate: 1.0e-4
  train_batch_size: 16
  max_train_steps: 100000
  mixed_precision: "bf16"
  checkpointing_steps: 1000
  visualization_steps: 500
  image_key: "edited_img"          # 只取 GT 图像，不取条件

model:
  target: lightningdit.tokenizer.vae32x_da.DAVAE32xFrom16x
  params:
    latent_channels_16x: 32
    latent_channels_32x: 128       # C=32 + D=96
    preconv_channels: 2048
    align_method: "mean"           # detail->teacher 对齐方式
    freeze_student_encoder: true   # 冻结原 16x encoder
    freeze_student_decoder: true   # 冻结原 16x decoder

loss:
  target: ldm.modules.losses.LPIPSWithDiscriminator
  params:
    disc_start: 5001               # 延迟启动 discriminator
    kl_weight: 1.0e-6
    pixelloss_weight: 1.0
    disc_weight: 0.1
    perceptual_weight: 1.0
    vf_weight: 0.5                 # 语义对齐权重
    vf_proj_use_mse: true          # 使用 MSE 对齐
```

---

## 2. 数据流

### 2.1 Dataloader

```python
# train_vae32x_from_edit_gt.py: load_multiratio_dataset
from common.flame.core.datasets.online_multi_ratio_instruction_sr import MultiRatio
```

沿用编辑训练的 `MultiRatio` dataloader，但**只取 `batch["edited_img"]`**，不加载：

- degraded image / reference image / mask
- prompt / text encoder
- timestep / noise / flow matching 相关数据

`collate_list` 把 batch 拼成 list，后续在 `stack_gt_images` 里转成 tensor。

### 2.2 图像预处理

```python
# train_vae32x_from_edit_gt.py: stack_gt_images
pixel_values = stack_gt_images(batch, args.image_key, device, dtype)
# 结果: [B, 3, H, W], 范围 [-1, 1]
```

模型内部还会做 `pad_multiple=32` 的 reflect padding，保证尺寸可被 32 整除。

---

## 3. 模型结构 (`DAVAE32xFrom16x`)

文件：`lightningdit/tokenizer/vae32x_da.py`

### 3.1 核心设计

32x DA-VAE 是**双路径**结构：

```
输入图像 x [B, 3, H, W]
    |
    ├── teacher path (冻结)
    │       resize 0.5x -> 冻结 16x VAE encoder -> z_base [B, 32, H/32, W/32]
    │
    └── student path (可训练新增模块)
            原 16x encoder 到 preconv16 [B, 2048, H/16, W/16]
            -> DCDownBlock2d(factor=2) -> detail moments [B, 192, H/32, W/32]
            -> DiagonalGaussianDistribution -> z_detail [B, 96, H/32, W/32]

z_detail --align--> z_detail_align [B, 32, H/32, W/32]
z32 = concat([z_base, z_detail]) [B, 128, H/32, W/32]

z32 -> DCUpBlock2d(factor=2) -> preconv16_hat [B, 2048, H/16, W/16]
    -> 原 16x decoder -> x_rec [B, 3, H, W]
```

### 3.2 关键组件

| 组件 | 类名 | 作用 |
|------|------|------|
| 额外 2x 下采样 | `DCDownBlock2d` | `preconv16` -> detail moments，含 shortcut |
| 额外 2x 上采样 | `DCUpBlock2d` | `z32` -> `preconv16_hat`，含 shortcut |
| 语义对齐头 | `_align_detail` | `mean`(默认): 96ch->32ch group mean；`proj`: 1x1 Conv |
| Teacher | 冻结 16x VAE | 输入半分辨率图，提供 `z_base` 对齐目标 |
| Student Encoder/Decoder | 原 16x VAE | 默认冻结，只训练新增的 Down/Up block |

### 3.3 初始化与冻结策略

```python
# __init__ 中
_load_state_dict_flexible(self.student, student_ckpt_path)
_load_state_dict_flexible(self.teacher, teacher_ckpt_path)

_freeze(self.teacher)                           # teacher 完全冻结
_freeze(self.student.encoder)                   # 默认冻结
_freeze(self.student.decoder)                   # 默认冻结
# 只有 extra_down / extra_up / align_proj(若用proj) 可训练
```

### 3.4 Forward 返回值

```python
recon, detail_post, extra = model(x, sample_posterior=True)

# extra 包含训练 loss 需要的全部中间变量：
extra = {
    "teacher_posterior": teacher_post,   # teacher 的 Gaussian posterior
    "z_teacher": z_base,                 # 对齐目标 [B,32,H/32,W/32]
    "z_base": z_base,
    "z_detail": z_detail,                # 原始 detail latent [B,96,H/32,W/32]
    "z_detail_align": z_detail_align,    # 对齐后的 detail [B,32,H/32,W/32]
    "z_combined": z32,                   # concat 后 [B,128,H/32,W/32]
    "z_student": z_detail,
    "z_student_align": z_detail_align,
}
```

---

## 4. 损失函数 (`LPIPSWithDiscriminator`)

文件：`lightningdit/davae/ldm/modules/losses/contperceptual.py`

### 4.1 总损失公式

```
L_total = L_rec + lambda_kl * L_kl + lambda_gan * L_gan + lambda_vf * L_align
```

### 4.2 各分项详解

#### 重建损失 `L_rec`

```python
rec_loss = |x - x_rec| + perceptual_weight * LPIPS(x, x_rec)
```

- `pixelloss_weight=1.0`：L1 像素重建
- `perceptual_weight=1.0`：LPIPS 感知损失（冻结 VGG 特征）
- `pp_style=true`：使用 mean 而非 sum 做归一化，训练更稳定

#### KL 损失 `L_kl`

```python
kl_loss = KL(q_detail(z_d|x) || N(0, I))
# weight: kl_weight = 1.0e-6
```

只作用于 student/detail posterior，teacher 不参与 KL。

#### GAN 损失 `L_gan`

- **Generator loss** (`optimizer_idx=0`)：
  ```python
  g_loss = -mean(discriminator(x_rec))
  d_weight = adaptive_weight(nll_loss, g_loss) * disc_weight
  ```
  `adaptive_weight` 通过梯度范数自动平衡重建和对抗损失的尺度。

- **Discriminator loss** (`optimizer_idx=1`)：
  ```python
  d_loss = hinge_loss(discriminator(x_real), discriminator(x_rec.detach()))
  ```
  Discriminator 在 `global_step >= disc_start=5001` 后才启动。

#### 语义对齐损失 `L_align`

```python
# 默认 vf_proj_use_mse=true, align_method="mean"
vf_loss = MSE(z_detail_align, z_teacher)
```

让对齐后的 detail latent 在语义结构上与 teacher latent 保持一致。

若 `vf_proj_use_mse=false`，则回退到论文原始的 cosine similarity + distance matrix 组合损失。

#### 可选 PatchEmbed 对齐 `L_pe`

```python
pe_loss = MSE(PE_student(z_detail), PE_teacher(z_teacher))
```

当前 Stage 1 未启用（`pe_align_enable=false`），留给 Stage 2 DiT 适配阶段使用。

### 4.3 自适应权重

```python
# calculate_adaptive_weight: 平衡 reconstruction vs GAN
d_weight = ||grad_nll|| / (||grad_g|| + 1e-4) * disc_weight

# calculate_adaptive_weight_vf: 平衡 reconstruction vs alignment
vf_weight = ||grad_nll|| / (||grad_vf|| + 1e-4) * vf_weight
```

当 `adaptive_vf=true` 时，语义对齐权重也是自适应的。

---

## 5. 训练循环 (Training Loop)

文件：`tools/train_vae32x_from_edit_gt.py:main()`

### 5.1 流程图

```
初始化 Accelerator + Model + Loss + Optimizers + Schedulers
    |
    v
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        |
        ├── Generator Step (accumulate)
        │       pixel_values = batch["edited_img"] -> [B,3,H,W]
        │       recon, posterior, extra = model(pixel_values, sample_posterior=True)
        │       ae_loss, ae_log = loss_module(
        │           inputs=pixel_values, reconstructions=recon,
        │           posteriors=posterior, optimizer_idx=0,
        │           global_step=global_step,
        │           last_layer=model.get_last_layer(),
        │           z=extra["z_detail_align"],
        │           aux_feature=extra["z_teacher"],
        │           enc_last_layer=model.get_encoder_last_layer(),
        │           z_pe=extra["z_detail"],
        │           align_method=model.align_method,
        │       )
        │       backward(ae_loss)
        │       clip_grad_norm(model.trainable_params)
        │       optimizer.step(); lr_scheduler.step()
        │
        ├── Discriminator Step (if started)
        │       disc_loss, disc_log = loss_module(
        │           inputs=pixel_values, reconstructions=recon.detach(),
        │           posteriors=posterior, optimizer_idx=1,
        │           global_step=global_step,
        │           last_layer=model.get_last_layer(),
        │       )
        │       backward(disc_loss)
        │       discriminator_optimizer.step(); disc_lr_scheduler.step()
        │
        └── Logging & Checkpointing
                global_step += 1
                log -> tensorboard: ae_loss, disc_loss, lr, rec_loss, kl_loss, vf_loss...
                
                if step % visualization_steps == 0:
                    save gt/rec grid to outputs/vae32x/visualization/
                
                if step % val_visualization_steps == 0:
                    model.eval(); run val dataloader; save val grid
                
                if step % checkpointing_steps == 0:
                    save checkpoint + vae32x_da_state_dict.pt
                    rotate_checkpoints()
```

### 5.2 关键注意点

| 项目 | 说明 |
|------|------|
| `sample_posterior=True` | train 时从 posterior 采样；val 时用 `mode()` |
| `optimizer_idx` | `0`=Generator (AE), `1`=Discriminator，交替更新 |
| `recon.detach()` | Discriminator 步骤中重建图要 detach，阻断梯度回传 |
| gradient clipping | `max_grad_norm=1.0`，只裁剪可训练参数 |
| mixed precision | 默认 `bf16`，model 用 weight_dtype，loss_module 保持 fp32 |

---

## 6. 分阶段训练策略

### Stage 0: 初始化

1. 加载已有 `f16c32` checkpoint 到 student 和 teacher。
2. 冻结 teacher、student encoder、student decoder。
3. 新增 `DCDown2d` / `DCUp2d` 随机初始化。
4. 默认 `align_method=mean`，无需额外对齐参数。

### Stage 1: 训练 32x VAE (当前实现)

- **目标**: 重建能力 + 语义对齐能力
- **数据**: 仅 `edited_img` GT 图像
- **可训练参数**: `extra_down`, `extra_up`, `align_proj`(若用 proj)
- **推荐超参**: lr=1e-4, batch=16~128, steps=100K, kl=1e-6, disc_start=5001, vf=0.5
- **监控指标**: `train/rec_loss`, `train/kl_loss`, `train/vf_loss`, `train/disc_loss`

### Stage 2: DiT/编辑模型适配 (未实现)

- 冻结 VAE，训练 DiT 输入适配层或 LoRA。
- 可引入 PatchEmbed 对齐 (`pe_align_enable=true`)。

---

## 7. 关键代码文件索引

| 文件 | 内容 |
|------|------|
| `configs/vae32x/train_vae32x_from_edit_gt.yaml` | 训练配置 |
| `tools/train_vae32x_from_edit_gt.py` | 训练主脚本 |
| `lightningdit/tokenizer/vae32x_da.py` | `DAVAE32xFrom16x` 模型定义 |
| `lightningdit/davae/ldm/modules/losses/contperceptual.py` | `LPIPSWithDiscriminator` 损失 |
| `docs/vae32x_architecture.md` | 网络架构设计文档 |

---

## 8. 快速启动检查清单

```bash
# 1. 确认分支
python -c "import torch; print(torch.__version__)"

# 2. 修改配置文件中的路径
#    - student_ckpt_path / teacher_ckpt_path: 指向现有 f16c32 VAE
#    - dataset_config_name / val_dataset_config_name: 指向 MultiRatio 配置

# 3. 启动训练
accelerate launch --mixed_precision bf16 tools/train_vae32x_from_edit_gt.py \
    --yml_path configs/vae32x/train_vae32x_from_edit_gt.yaml

# 4. 监控 tensorboard
tensorboard --logdir outputs/vae32x/logs
```

---

*文档生成时间: 2026-05-27, 基于 da-vae-32x-pretrain 分支最新代码。*
