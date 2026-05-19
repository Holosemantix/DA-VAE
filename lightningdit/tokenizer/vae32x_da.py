from __future__ import annotations

import copy
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _ensure_ldm_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    davae_root = repo_root / "davae"
    if str(davae_root) not in sys.path:
        sys.path.insert(0, str(davae_root))


_ensure_ldm_on_path()
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution  # noqa: E402


def _get_obj_from_str(path: str):
    module, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def instantiate_from_config(config: Dict[str, Any]):
    if config is None or "target" not in config:
        raise KeyError("Expected config with key `target`.")
    params = config.get("params", {}) or {}
    return _get_obj_from_str(config["target"])(**params)


def _load_state_dict_flexible(module: nn.Module, ckpt_path: Optional[str], strict: bool = False) -> None:
    if not ckpt_path:
        return
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "ema"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")

    cleaned = {}
    for key, value in obj.items():
        key = key[7:] if key.startswith("module.") else key
        if key.startswith("model."):
            cleaned[key[len("model."):]] = value
        else:
            cleaned[key] = value

    missing, unexpected = module.load_state_dict(cleaned, strict=strict)
    print(
        f"[DAVAE32x] loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print(f"[DAVAE32x] missing keys sample: {missing[:20]}")
    if unexpected:
        print(f"[DAVAE32x] unexpected keys sample: {unexpected[:20]}")


def _unwrap_autoencoder(model: nn.Module) -> nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "encoder") and hasattr(model.model, "decoder"):
        return model.model
    return model


def _freeze(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


class DCDownBlock2d(nn.Module):
    """DA-style local 2x compression block for Gaussian moments."""

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2, shortcut: bool = True) -> None:
        super().__init__()
        self.factor = int(factor)
        self.shortcut = bool(shortcut)
        ratio = self.factor ** 2
        if out_channels % ratio != 0:
            raise ValueError(f"out_channels={out_channels} must be divisible by factor^2={ratio}")

        self.conv = nn.Conv2d(in_channels, out_channels // ratio, kernel_size=3, stride=1, padding=1)
        total_shortcut_channels = in_channels * ratio
        self._divisible = total_shortcut_channels % out_channels == 0
        if self.shortcut and self._divisible:
            self.group_size = total_shortcut_channels // out_channels
            self.shortcut_proj = None
        elif self.shortcut:
            self.group_size = None
            self.shortcut_proj = nn.Conv2d(total_shortcut_channels, out_channels, kernel_size=1, bias=False)
            nn.init.zeros_(self.shortcut_proj.weight)
        else:
            self.group_size = None
            self.shortcut_proj = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        height, width = hidden_states.shape[-2:]
        if height % self.factor != 0 or width % self.factor != 0:
            raise ValueError(f"Input spatial size {(height, width)} must be divisible by {self.factor}")

        x = self.conv(hidden_states)
        x = F.pixel_unshuffle(x, self.factor)
        if not self.shortcut:
            return x

        y = F.pixel_unshuffle(hidden_states, self.factor)
        if self._divisible:
            y = y.unflatten(1, (-1, self.group_size)).mean(dim=2)
        else:
            y = self.shortcut_proj(y)
        return x + y


class DCUpBlock2d(nn.Module):
    """Inverse 2x expansion block before the original 16x decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
        interpolate: bool = False,
    ) -> None:
        super().__init__()
        self.factor = int(factor)
        self.shortcut = bool(shortcut)
        self.interpolate = bool(interpolate)
        self.interpolation_mode = interpolation_mode
        ratio = self.factor ** 2

        if self.shortcut and (out_channels * ratio) % in_channels != 0:
            raise ValueError(
                f"out_channels * factor^2 must be divisible by in_channels; got "
                f"{out_channels} * {ratio} vs {in_channels}"
            )
        self.repeats = (out_channels * ratio) // in_channels if self.shortcut else 0
        conv_out = out_channels if self.interpolate else out_channels * ratio
        self.conv = nn.Conv2d(in_channels, conv_out, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if not self.shortcut:
            return x

        y = hidden_states.repeat_interleave(
            self.repeats,
            dim=1,
            output_size=hidden_states.shape[1] * self.repeats,
        )
        y = F.pixel_shuffle(y, self.factor)
        return x + y


class DAVAE32xFrom16x(nn.Module):
    """32x DA-VAE wrapper around the provided 16x Swin KL autoencoder.

    The trainable student path encodes the full-resolution image, compresses
    the 16x Gaussian moments by another factor of 2, and decodes by expanding
    the sampled 32x latent back to the original 16x decoder input.

    The frozen teacher path encodes a half-resolution image with the existing
    16x VAE, producing a spatially matched `H/32 x W/32` semantic latent.
    """

    def __init__(
        self,
        student_config: Dict[str, Any],
        teacher_config: Optional[Dict[str, Any]] = None,
        student_ckpt_path: Optional[str] = None,
        teacher_ckpt_path: Optional[str] = None,
        latent_channels_16x: int = 32,
        latent_channels_32x: int = 128,
        teacher_latent_channels: int = 32,
        preconv_channels: int = 2048,
        align_method: str = "mean",
        pad_multiple: int = 32,
        teacher_downsample_mode: str = "bicubic",
        strict_load: bool = False,
        freeze_student_encoder: bool = True,
        freeze_student_decoder: bool = True,
    ) -> None:
        super().__init__()
        if align_method not in ("mean", "proj"):
            raise ValueError(f"align_method must be 'mean' or 'proj', got {align_method}")

        self.student = _unwrap_autoencoder(instantiate_from_config(student_config))
        if teacher_config is None:
            teacher_config = copy.deepcopy(student_config)
        self.teacher = _unwrap_autoencoder(instantiate_from_config(teacher_config))

        _load_state_dict_flexible(self.student, student_ckpt_path, strict=strict_load)
        _load_state_dict_flexible(self.teacher, teacher_ckpt_path or student_ckpt_path, strict=strict_load)

        _freeze(self.teacher)
        if freeze_student_encoder and hasattr(self.student, "encoder"):
            _freeze(self.student.encoder)
        if freeze_student_decoder and hasattr(self.student, "decoder"):
            _freeze(self.student.decoder)

        self.latent_channels_16x = int(latent_channels_16x)
        self.latent_channels_32x = int(latent_channels_32x)
        self.teacher_latent_channels = int(teacher_latent_channels)
        self.preconv_channels = int(preconv_channels)
        self.align_method = align_method
        self.pad_multiple = int(pad_multiple)
        self.teacher_downsample_mode = teacher_downsample_mode

        self.extra_down = DCDownBlock2d(
            in_channels=self.preconv_channels,
            out_channels=2 * self.latent_channels_32x,
            factor=2,
            shortcut=True,
        )
        self.extra_up = DCUpBlock2d(
            in_channels=self.latent_channels_32x,
            out_channels=self.preconv_channels,
            factor=2,
            shortcut=True,
        )

        if self.align_method == "proj":
            self.align_proj = nn.Conv2d(self.latent_channels_32x, self.teacher_latent_channels, 1)
        else:
            self.align_proj = None
            if self.latent_channels_32x % self.teacher_latent_channels != 0:
                raise ValueError(
                    "mean alignment requires latent_channels_32x to be divisible by "
                    f"teacher_latent_channels, got {self.latent_channels_32x} and {self.teacher_latent_channels}"
                )

        self.config = SimpleNamespace(
            latent_channels=self.latent_channels_32x,
            scaling_factor=getattr(getattr(self.student, "config", None), "scaling_factor", 1.0),
            shift_factor=getattr(getattr(self.student, "config", None), "shift_factor", 0.0),
            block_out_channels=[0, 0, 0, 0, 0, 0],
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _pad_image(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        org_h, org_w = x.shape[-2:]
        target_h = ((org_h + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        target_w = ((org_w + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        pad_h = target_h - org_h
        pad_w = target_w - org_w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (org_h, org_w)

    def _encode_16x_moments(self, ae: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(ae, "encoder"):
            raise AttributeError("The wrapped 16x VAE must expose an `encoder` module.")
        return ae.encoder(x)

    def _encode_student_preconv(self, x: torch.Tensor) -> torch.Tensor:
        """Run the provided 16x encoder up to the feature before conv_out.

        This mirrors DA-VAE's insertion point: after encoder norm/activation and
        before the Gaussian moments projection. For the provided Swin VAE this
        is the 2x patchified high-channel feature at 16x spatial stride.
        """
        if not hasattr(self.student, "encoder"):
            raise AttributeError("The wrapped 16x VAE must expose an `encoder` module.")
        e = self.student.encoder
        temb = None

        h = e.conv_in(x)
        for i_level in range(e.num_resolutions):
            for i_block in range(e.num_res_blocks[i_level]):
                h = e.down[i_level].block[i_block](h, temb)
                if len(e.down[i_level].attn) > 0:
                    h = e.down[i_level].attn[i_block](h)
            if i_level != e.num_resolutions - 1:
                h = e.down[i_level].downsample(h)

        h = e.mid.block_1(h, temb)
        h = e.mid.attn_1(h)
        h = e.mid.block_2(h, temb)
        h = rearrange(h, "b c (h dh) (w dw) -> b (c dh dw) h w", dh=2, dw=2)
        h = e.norm_out(h)
        h = F.silu(h)
        return h

    def _decode_from_student_preconv(self, h: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.student, "decoder"):
            raise AttributeError("The wrapped 16x VAE must expose a `decoder` module.")
        d = self.student.decoder
        temb = None

        h = rearrange(h, "b (c dh dw) h w -> b c (h dh) (w dw)", dh=2, dw=2)

        h = d.mid.block_1(h, temb)
        h = d.mid.attn_1(h)
        h = d.mid.block_2(h, temb)

        for i_level in reversed(range(d.num_resolutions)):
            for i_block in range(d.num_res_blocks[i_level] + 1):
                h = d.up[i_level].block[i_block](h, temb)
                if len(d.up[i_level].attn) > 0:
                    h = d.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = d.up[i_level].upsample(h)

        if getattr(d, "give_pre_end", False):
            return h

        h = d.norm_out(h)
        h = F.silu(h)
        h = d.conv_out(h)
        if getattr(d, "tanh_out", False):
            h = torch.tanh(h)
        return h

    def _align_student(self, z32: torch.Tensor) -> torch.Tensor:
        if self.align_method == "proj":
            return self.align_proj(z32)
        bsz, channels, height, width = z32.shape
        groups = channels // self.teacher_latent_channels
        return z32.view(bsz, self.teacher_latent_channels, groups, height, width).mean(dim=2)

    @torch.no_grad()
    def encode_teacher(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        x, _ = self._pad_image(x)
        x_down = F.interpolate(
            x,
            scale_factor=0.5,
            mode=self.teacher_downsample_mode,
            align_corners=False if self.teacher_downsample_mode in ("bilinear", "bicubic") else None,
        )
        moments = self._encode_16x_moments(self.teacher, x_down)
        return DiagonalGaussianDistribution(moments)

    def encode_student(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        x, _ = self._pad_image(x)
        preconv16 = self._encode_student_preconv(x)
        moments32 = self.extra_down(preconv16)
        return DiagonalGaussianDistribution(moments32)

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        posterior = self.encode_student(x)
        if not return_dict:
            return (posterior,)
        return SimpleNamespace(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        preconv16 = self.extra_up(z)
        dec = self._decode_from_student_preconv(preconv16)
        if not return_dict:
            return (dec,)
        return dec

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        x_pad, (org_h, org_w) = self._pad_image(x)
        with torch.no_grad():
            teacher_post = self.encode_teacher(x)
            z_teacher = teacher_post.mode().detach()

        student_post = self.encode_student(x_pad)
        z32 = student_post.sample() if sample_posterior else student_post.mode()
        z32_align = self._align_student(z32)

        recon = self.decode(z32, return_dict=False)[0]
        recon = recon[..., :org_h, :org_w]
        return recon, student_post, {
            "teacher_posterior": teacher_post,
            "z_teacher": z_teacher,
            "z_student": z32,
            "z_student_align": z32_align,
        }

    def get_last_layer(self):
        if hasattr(self.student, "decoder") and hasattr(self.student.decoder, "conv_out"):
            return self.student.decoder.conv_out.weight
        return self.extra_up.conv.weight

    def get_encoder_last_layer(self):
        return self.extra_down.conv.weight

    def save_pretrained_weight(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "vae32x_da.pt")
        torch.save(self.state_dict(), path)
        return path
