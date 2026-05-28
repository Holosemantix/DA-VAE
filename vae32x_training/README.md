# 32x DA-VAE 独立训练包

这是从 `DA-VAE/da-vae-32x-pretrain` 分支抽取的独立训练代码，可直接拷贝到自己的训练工程里使用。

---

## 目录结构

```
vae32x_training/
├── train.py                          # 训练入口
├── train_vae32x_from_edit_gt.yaml    # 默认配置文件
├── debug_npu.sh                      # NPU 本地调试脚本
├── README.md                         # 本文件
├── models/
│   └── vae32x_da.py                  # DAVAE32xFrom16x 模型
├── losses/
│   └── contperceptual.py             # LPIPSWithDiscriminator 损失
├── utils/
│   └── common.py                     # 通用工具 (instantiate_from_config, DiagonalGaussianDistribution)
└── third_party/
    └── taming/                       # taming-transformers 依赖（精简版）
        ├── util.py
        └── modules/
            ├── util.py
            ├── losses/
            │   ├── vqperceptual.py
            │   └── lpips.py
            └── discriminator/
                └── model.py
```

---

## 前置依赖

```bash
pip install torch torchvision torch-npu  # NPU 环境
pip install accelerate einops pyyaml easydict diffusers
pip install tqdm requests                # taming lpips 下载预训练权重用
```

> **注意**：本包仍然依赖你已有的 **16x VAE 模型定义**（如 `f16c32_vae_swin_dinov3_dec.autoencoder_2d_16x_5layer_ResBlockList_swin_2d.AutoencoderKL`）。你需要确保该模块在 Python 路径中，或在 `train_vae32x_from_edit_gt.yaml` 的 `student_config.target` / `teacher_config.target` 里指向你自己的 AutoencoderKL 类。

---

## 快速开始

### 1. 修改配置文件

打开 `train_vae32x_from_edit_gt.yaml`，修改以下路径：

```yaml
args:
  output_dir: "your_output_dir"
  dataset_config_name: "/path/to/train_dataset.yaml"      # MultiRatio 配置
  val_dataset_config_name: "/path/to/val_dataset.yaml"    # 可选

model:
  params:
    student_ckpt_path: "/path/to/your_f16c32_vae.pth"     # 16x VAE 权重
    teacher_ckpt_path: "/path/to/your_f16c32_vae.pth"     # 通常和 student 相同
    student_config:
      target: your_package.your_vae.AutoencoderKL           # 指向你的 16x VAE 类
    teacher_config:
      target: your_package.your_vae.AutoencoderKL           # 通常和 student 相同
```

### 2. NPU 本地调试（dummy dataset）

```bash
bash debug_npu.sh
```

这会使用随机生成的 dummy 数据跑 20 步，验证：
- 模型前向/反向
- 损失计算
- checkpoint 保存
- 可视化输出

### 3. 正式训练

```bash
python3 train.py \
    --yml_path train_vae32x_from_edit_gt.yaml \
    --train_batch_size 16 \
    --max_train_steps 100000 \
    --mixed_precision bf16
```

分布式训练：

```bash
accelerate launch --mixed_precision bf16 train.py \
    --yml_path train_vae32x_from_edit_gt.yaml
```

---

## 关键设计说明

### 可训练参数

默认只有新增的模块参与训练：

- `extra_down` (`DCDownBlock2d`): 32x detail encoder
- `extra_up` (`DCUpBlock2d`): 32x detail decoder
- `align_proj` (当 `align_method="proj"` 时)

原 16x VAE 的 encoder/decoder 和 teacher path **全部冻结**。

### 双路径结构

| 路径 | 输入 | 输出 | 是否可训练 |
|------|------|------|-----------|
| Teacher | 半分辨率图 | `z_base [B,32,H/32,W/32]` | 冻结 |
| Student | 原分辨率图 | `z_detail [B,96,H/32,W/32]` | 可训练（仅新增层） |

最终 latent：`z32 = concat([z_base, z_detail])` → `[B, 128, H/32, W/32]`

### 损失组成

```
L_total = L_rec(L1 + LPIPS) + λ_kl * L_kl + λ_gan * L_gan + λ_vf * L_align(MSE)
```

- `disc_start=5001`: discriminator 延迟启动
- `vf_proj_use_mse=true`: 语义对齐使用 MSE（更稳定）

---

## 迁移到自有工程

直接把整个 `vae32x_training/` 文件夹拷贝到你的项目里即可。需要保证：

1. 你的 Python 环境能 import `torch_npu`（NPU）。
2. 你的工程里已有 16x VAE 的模型类定义，并在 YAML 中正确配置 `target`。
3. 你的 dataloader 需要返回 `{"edited_img": tensor}` 格式的 batch。

---

*基于 DA-VAE da-vae-32x-pretrain 分支抽取。*
