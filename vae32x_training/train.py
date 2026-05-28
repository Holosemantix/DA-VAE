#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, is_compiled_module
from diffusers.optimization import get_scheduler
from easydict import EasyDict

try:
    import moxing as mox
except Exception:
    mox = None

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except Exception:
    pass


from utils.common import instantiate_from_config


logger = get_logger(__name__)


def read_yaml(path: str) -> EasyDict:
    with open(path, "r") as handle:
        return EasyDict(yaml.safe_load(handle))


def merge_args_with_yaml(args: EasyDict) -> EasyDict:
    if not args.yml_path:
        return args
    cfg = read_yaml(args.yml_path)
    for key, value in cfg.get("args", {}).items():
        if key in ("train_url", "resume_from_checkpoint") and getattr(args, key, None):
            continue
        args[key] = value
    args.model = cfg.model
    args.loss = cfg.loss
    return args


def parse_args(input_args: Optional[List[str]] = None) -> EasyDict:
    parser = argparse.ArgumentParser("Train 32x DA-VAE from edit dataloader GT images.")
    parser.add_argument("--yml_path", type=str, required=True)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--val_dataset_config_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/vae32x")
    parser.add_argument("--train_url", type=str, default=None)
    parser.add_argument("--image_key", type=str, default="edited_img")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--discriminator_learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1.0e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--visualization_steps", type=int, default=500)
    parser.add_argument("--val_visualization_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument("--dummy_dataset", action="store_true", default=False)
    parsed = parser.parse_args(input_args)
    args = EasyDict(vars(parsed))
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return merge_args_with_yaml(args)


def set_seed(seed: int) -> None:
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_multiratio_dataset(config_path: str, batch_size: int):
    try:
        from common.flame.core.datasets.online_multi_ratio_instruction_sr import MultiRatio
    except ImportError as exc:
        raise ImportError(
            "Failed to import MultiRatio dataset from 'common.flame'. "
            "Please ensure your project's dataloader module is in PYTHONPATH, "
            "or use --dummy_dataset for local debugging. "
            f"Original error: {exc}"
        ) from exc

    with open(config_path, "r") as handle:
        cfg = yaml.safe_load(handle)
    cfg["rank"] = int(os.getenv("RANK", "0"))
    cfg["world_size"] = int(os.getenv("WORLD_SIZE", "1"))
    cfg["batch_size"] = batch_size
    return MultiRatio(cfg)


class DummyImageDataset(torch.utils.data.Dataset):
    def __init__(self, length: int = 32, size: int = 256, key: str = "edited_img") -> None:
        self.length = length
        self.size = size
        self.key = key

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {self.key: torch.rand(3, self.size, self.size) * 2 - 1}


def collate_list(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {key: [item[key] for item in batch] for key in batch[0].keys()}


def make_dataloader(args: EasyDict, train: bool = True):
    if args.dummy_dataset:
        dataset = DummyImageDataset(key=args.image_key)
    else:
        cfg_path = args.dataset_config_name if train else args.val_dataset_config_name
        if not cfg_path:
            raise ValueError("dataset_config_name/val_dataset_config_name must be set unless --dummy_dataset is used.")
        dataset = load_multiratio_dataset(cfg_path, args.train_batch_size)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.train_batch_size if train else 1,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_list,
        persistent_workers=args.dataloader_num_workers > 0,
    )


def stack_gt_images(batch: Dict[str, Any], image_key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if image_key not in batch:
        available = ", ".join(batch.keys())
        raise KeyError(f"GT image key `{image_key}` not found in batch. Available keys: {available}")
    values = batch[image_key]
    if isinstance(values, torch.Tensor):
        images = values
    else:
        images = torch.stack(values)
    if images.ndim != 4:
        raise ValueError(f"Expected image tensor with shape B,C,H,W, got {tuple(images.shape)}")
    return images.to(device=device, dtype=dtype, non_blocking=True)


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def maybe_copy_to_remote(local_path: str, remote_path: Optional[str]) -> None:
    if not remote_path or mox is None:
        return
    try:
        if os.path.isdir(local_path):
            mox.file.copy_parallel(local_path, remote_path)
        else:
            mox.file.copy(local_path, remote_path)
    except Exception as exc:
        logger.warning(f"remote copy failed: {local_path} -> {remote_path}: {exc}")


def gather_loss(loss: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        gathered = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, loss.detach())
        return torch.stack(gathered).mean()
    return loss.detach()


def save_visualization(
    accelerator: Accelerator,
    args: EasyDict,
    images: Dict[str, torch.Tensor],
    global_step: int,
    subdir: str,
    nrow: int = 4,
) -> None:
    vis_dir = os.path.join(args.output_dir, subdir)
    gathered = {}
    for key, value in images.items():
        value = torch.clamp((value.float() + 1.0) / 2.0, 0.0, 1.0)
        gathered[key] = accelerator.gather(value.contiguous())

    if accelerator.is_main_process:
        if os.path.exists(vis_dir):
            shutil.rmtree(vis_dir)
        os.makedirs(vis_dir, exist_ok=True)
        for key, value in gathered.items():
            grid = torchvision.utils.make_grid(value, nrow=nrow)
            torchvision.utils.save_image(grid, os.path.join(vis_dir, f"iter_{global_step}_{key}.jpg"))
        maybe_copy_to_remote(vis_dir, os.path.join(args.train_url, subdir) if args.train_url else None)


def rotate_checkpoints(args: EasyDict) -> None:
    if not args.checkpoints_total_limit or not os.path.isdir(args.output_dir):
        return
    checkpoints = [
        item for item in os.listdir(args.output_dir)
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, item))
    ]
    checkpoints = sorted(checkpoints, key=lambda name: int(name.split("-")[-1]))
    excess = len(checkpoints) - args.checkpoints_total_limit
    for ckpt in checkpoints[:max(0, excess)]:
        shutil.rmtree(os.path.join(args.output_dir, ckpt), ignore_errors=True)


def main(args: EasyDict) -> None:
    if args.train_url and args.train_url.endswith("/"):
        args.train_url = args.train_url[:-1]
    if args.train_url:
        base_name = os.path.basename(args.train_url)
        dirname = os.path.dirname(args.train_url)
        prefix = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        args.train_url = os.path.join(dirname, prefix + "_" + base_name)

    logging_dir = Path(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed + max(args.local_rank, 0))

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)

    model = instantiate_from_config(args.model)
    loss_module = instantiate_from_config(args.loss)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    model.to(accelerator.device, dtype=weight_dtype)
    loss_module.to(accelerator.device)

    ae_params = [p for p in model.parameters() if p.requires_grad]
    if hasattr(loss_module, "logvar"):
        ae_params.append(loss_module.logvar)
    optimizer = torch.optim.AdamW(
        ae_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    discriminator_optimizer = None
    if hasattr(loss_module, "discriminator") and args.discriminator_learning_rate > 0:
        discriminator_optimizer = torch.optim.AdamW(
            loss_module.discriminator.parameters(),
            lr=args.discriminator_learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    train_dataloader = make_dataloader(args, train=True)
    val_dataloader = make_dataloader(args, train=False) if args.val_dataset_config_name or args.dummy_dataset else None

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    discriminator_lr_scheduler = None
    if discriminator_optimizer is not None:
        discriminator_lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=discriminator_optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    prepare_items = [model, loss_module, optimizer, train_dataloader, lr_scheduler]
    if discriminator_optimizer is not None:
        prepare_items.extend([discriminator_optimizer, discriminator_lr_scheduler])
    prepared = accelerator.prepare(*prepare_items)
    model, loss_module, optimizer, train_dataloader, lr_scheduler = prepared[:5]
    if discriminator_optimizer is not None:
        discriminator_optimizer, discriminator_lr_scheduler = prepared[5:]
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("vae32x-da", config=dict(args))

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        if resume_path == "latest":
            candidates = [
                item for item in os.listdir(args.output_dir)
                if item.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, item))
            ]
            candidates = sorted(candidates, key=lambda name: int(name.split("-")[-1]))
            resume_path = os.path.join(args.output_dir, candidates[-1]) if candidates else None
        if resume_path:
            accelerator.load_state(resume_path)
            global_step = int(os.path.basename(resume_path).split("-")[-1])
            first_epoch = global_step // num_update_steps_per_epoch

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running 32x DA-VAE Training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    cfg_copied = False
    model.train()
    loss_module.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            iter_start = time.time()
            with accelerator.accumulate(model):
                pixel_values = stack_gt_images(batch, args.image_key, accelerator.device, weight_dtype)
                recon, posterior, extra = model(pixel_values, sample_posterior=True)

                ae_loss, ae_log = loss_module(
                    pixel_values.float(),
                    recon.float(),
                    posterior,
                    0,
                    global_step,
                    last_layer=unwrap_model(accelerator, model).get_last_layer(),
                    split="train",
                    z=extra["z_detail_align"],
                    aux_feature=extra["z_teacher"],
                    enc_last_layer=unwrap_model(accelerator, model).get_encoder_last_layer(),
                    z_pe=extra["z_detail"],
                    align_method=getattr(unwrap_model(accelerator, model), "align_method", "mean"),
                )

                optimizer.zero_grad()
                accelerator.backward(ae_loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

                disc_loss = None
                disc_log = {}
                if discriminator_optimizer is not None:
                    disc_loss, disc_log = loss_module(
                        pixel_values.float(),
                        recon.detach().float(),
                        posterior,
                        1,
                        global_step,
                        last_layer=unwrap_model(accelerator, model).get_last_layer(),
                        split="train",
                    )
                    discriminator_optimizer.zero_grad()
                    accelerator.backward(disc_loss)
                    discriminator_optimizer.step()
                    if discriminator_lr_scheduler is not None:
                        discriminator_lr_scheduler.step()

            if accelerator.sync_gradients:
                global_step += 1
                mean_ae_loss = gather_loss(ae_loss)
                logs = {
                    "train/aeloss": mean_ae_loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                }
                if disc_loss is not None:
                    logs["train/discloss"] = gather_loss(disc_loss).item()
                for source in (ae_log, disc_log):
                    for key, value in source.items():
                        if torch.is_tensor(value):
                            logs[key] = value.detach().float().mean().item()
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process and global_step % 10 == 0:
                    elapsed_ms = (time.time() - iter_start) * 1000
                    msg = (
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"iteration {global_step} / {args.max_train_steps} "
                        f"| lr: {lr_scheduler.get_last_lr()[0]:.2e} "
                        f"| elapsed ms: {elapsed_ms:.1f} "
                        f"| ae_loss: {mean_ae_loss.item():.5f}"
                    )
                    if disc_loss is not None:
                        msg += f" | disc_loss: {logs['train/discloss']:.5f}"
                    print(msg)

                if global_step == 1 or global_step % args.visualization_steps == 0:
                    with torch.no_grad():
                        save_visualization(
                            accelerator,
                            args,
                            {"gt": pixel_values.detach(), "rec": recon.detach()},
                            global_step,
                            "visualization",
                        )

                if val_dataloader is not None and (global_step == 1 or global_step % args.val_visualization_steps == 0):
                    model.eval()
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_pixels = stack_gt_images(val_batch, args.image_key, accelerator.device, weight_dtype)
                            val_recon, _, _ = model(val_pixels, sample_posterior=False)
                            save_visualization(
                                accelerator,
                                args,
                                {"gt": val_pixels.detach(), "rec": val_recon.detach()},
                                global_step,
                                "val_visualization",
                            )
                            break
                    model.train()

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        unwrapped = unwrap_model(accelerator, model)
                        torch.save(accelerator.get_state_dict(model), os.path.join(save_path, "vae32x_da_state_dict.pt"))
                        if hasattr(unwrapped, "save_pretrained_weight"):
                            unwrapped.save_pretrained_weight(save_path)
                        if not cfg_copied:
                            maybe_copy_to_remote(args.yml_path, os.path.join(args.train_url, os.path.basename(args.yml_path)) if args.train_url else None)
                            cfg_copied = True
                        maybe_copy_to_remote(save_path, os.path.join(args.train_url, f"checkpoint-{global_step}") if args.train_url else None)
                        rotate_checkpoints(args)
                    accelerator.wait_for_everyone()

                if global_step % 500 == 0:
                    gc.collect()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        torch.save(accelerator.get_state_dict(model), os.path.join(final_dir, "vae32x_da_state_dict.pt"))
        maybe_copy_to_remote(final_dir, os.path.join(args.train_url, "final") if args.train_url else None)
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
