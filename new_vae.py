import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf
# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from loguru import logger


def nonlinearity(x):
    # swish
    #return x*torch.sigmoid(x)
    return tnf.silu(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("PPPPPPPPPixel unshuffle: in averaging shortcut, ori  x shape:", x.shape)
        x = tnf.pixel_unshuffle(x, self.factor)
        # print("we are at here when x.shape length is 4::::x resized(unshuffle shape): ", x.shape)
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        # print("2d x shape after view", x.shape )
        x = x.mean(dim=2)
        # print("2d x shape after mean:", x.shape, " shortcut forward ends")
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer_convout(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("in averaging shortcut, ori  x shape:", x.shape)
        # x = tnf.pixel_unshuffle(x, self.factor)
        # print("x resized(unshuffle shape): ", x.shape)
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        # print("x shape after view", x.shape )
        x = x.mean(dim=2)
        # print("x shape after mean:", x.shape, " shortcut forward ends")
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer_2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("CCCCCCCCChannel duplicate: in duplicate shortcut, ori  x shape:", x.shape)
        x = x.repeat_interleave(self.repeats, dim=1)
        # print("x repeat shape: ", x.shape)
        x = tnf.pixel_shuffle(x, self.factor)
        # print("x shape after shuffle:", x.shape, " shortcut forward ends")
        return x


class Encode_conv_out_shortcut(nn.Module):
    def __init__(self, in_channels, z_channels, double_z=True):
        super().__init__()
        self.double_z = double_z
        self.z_channels = z_channels
        self.conv = torch.nn.Conv2d(in_channels,
                                    2*z_channels if double_z else z_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)  

        self.shortcut = PixelUnshuffleChannelAveragingDownSampleLayer_convout(in_channels, 2*z_channels if double_z else z_channels, factor=1)

    def forward(self, x):
        # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVvery important x shape:", x.shape)
        sh = self.shortcut(x)
        # x = tnf.pad(x, (1,1,1,1,2,0), mode="constant", value=0)
        # print(x.shape)
        x = self.conv(x)
        # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVvery important:", sh.shape, x.shape)
        x = x + sh
        return x



class Decode_conv_in_shortcut(nn.Module):
    def __init__(self, in_channels, z_channels, double_z=True):
        super().__init__()
        self.double_z = double_z
        self.z_channels = z_channels
        self.conv = torch.nn.Conv2d(z_channels,
                                       in_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.shortcut = ChannelDuplicatingPixelUnshuffleUpSampleLayer_2d(z_channels, in_channels, factor=1)

    def forward(self, x):
        # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVvery important x shape:", x.shape)
        sh = self.shortcut(x)
        # x = tnf.pad(x, (1,1,1,1,2,0), mode="constant", value=0)
        # print(x.shape)
        x = self.conv(x)
        # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVvery important:", sh.shape, x.shape)
        x = x + sh
        return x


class Upsample_shortcut(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.shortcut = ChannelDuplicatingPixelUnshuffleUpSampleLayer_2d(in_channels, in_channels, factor=2)

    def forward(self, x):
        # print("UUUUUUUUUUUUUpsample22222d: before any interpolation shape: ", x.shape)
        sh = self.shortcut(x)
        x = tnf.interpolate(x, scale_factor=2.0, mode="nearest")
        # print("after frist interpo", x.shape)
        if self.with_conv:
            x = self.conv(x) + sh
            # print("after conv ", x.shape)
        return x


class Downsample_shortcut(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
        self.shortcut = PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, in_channels, factor=2)

    def forward(self, x):
        if self.with_conv:
            sh = self.shortcut(x)
            pad = (0,1,0,1)
            x = tnf.pad(x, pad, mode="constant", value=0)
            x = self.conv(x) + sh
        else:
            x = tnf.avg_pool2d(x, kernel_size=2, stride=2) + self.shortcut(x)
        return x


class Downsample_shortcut_enlarge_ch(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels*2,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
        self.shortcut = PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, in_channels*2, factor=2)

    def forward(self, x):
        if self.with_conv:
            sh = self.shortcut(x)
            pad = (0,1,0,1)
            x = tnf.pad(x, pad, mode="constant", value=0)
            x = self.conv(x) + sh
        else:
            x = tnf.avg_pool2d(x, kernel_size=2, stride=2) + self.shortcut(x)
        return x



class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = tnf.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_




def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla"], f'attn_type {attn_type} unknown'
    return AttnBlock(in_channels)


class Encoder(nn.Module):
    def __init__(self, z_channels, resolution, in_channels=3, ch=128, attn_resolutions=[],
                 dropout=0.0, resamp_with_conv=True, attn_type="vanilla", **ignore_kwargs):
        super().__init__()
        logger.debug(f"Encoder(): {ignore_kwargs=}")

        self.ch = ch
        self.temb_ch = 0
        ch_mult = [1, 2, 4, 4, 4]
        num_res_blocks = [2, 3, 2, 2, 2]
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                if i_level==1 and i_block==0:
                    block_in = 256
                if i_level==2 and i_block==0:
                    block_in = 512
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level == 0 or i_level == 1:
                #### first two downsample could enlarge the channel on the same layer!!!! Same for upsample?
                down.downsample = Downsample_shortcut_enlarge_ch(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            elif i_level != self.num_resolutions-1:
                #### first two downsample could enlarge the channel on the same layer!!!! Same for upsample?
                down.downsample = Downsample_shortcut(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            # elif num_resize>3:
            #     down.downsample = Downsample_shortcut(block_in, resamp_with_conv)

            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = Encode_conv_out_shortcut(in_channels=block_in, z_channels=z_channels)


    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                #hs.append(h)
            if i_level != self.num_resolutions-1:
                h= self.down[i_level].downsample(h)

        # middle
        #h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, z_channels, resolution, ch=128, in_channels=3,
                 attn_resolutions=[], dropout=0.0, resamp_with_conv=True,
                 give_pre_end=False, tanh_out=False, attn_type="vanilla", **ignore_kwargs):
        super().__init__()
        logger.debug(f"Decoder(): {ignore_kwargs=}")

        self.ch = ch
        self.temb_ch = 0
        ch_mult = [1,2,4,4,4]
        num_res_blocks = [2,3,2,2,2]
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)

        # z to block_in
        self.conv_in = Decode_conv_in_shortcut(in_channels=block_in, z_channels=z_channels)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample_shortcut(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            # if num_resize>3:
            #     up.upsample = Upsample_shortcut(block_in, resamp_with_conv)
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None
        # print('dec', z.dtype)
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level]+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class AutoencoderKL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)

        freeze_enc = config.get("freeze_enc", False)
        if freeze_enc:
            logger.info(f"Freezing encoder parameters for AutoencoderKL() in {__file__}...")
            self.encoder.requires_grad_(False)

        self.enc_learnable = (not freeze_enc)

        self.z_mean = -0.02453034371137619
        self.z_std = 0.7610968947410583
        self.downsampling_factor = 16

    @staticmethod
    def gaussian_sample(mean, logvar):
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def encode(self, x):
        with torch.set_grad_enabled(mode=self.enc_learnable):
            qz_params = self.encoder(x)

        mean, logvar = torch.chunk(qz_params, chunks=2, dim=1)
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)

        mean, logvar = mean.float(), logvar.float()
        z = self.gaussian_sample(mean, logvar)

        x_rec = self.decode(z.to(dtype=x.dtype))
        return {"x_rec": x_rec, "z": z, "mean": mean, "logvar": logvar}


def majie_16x64ch(pretrained=None):
    config = {"z_channels": 64, "resolution": 256}
    model = AutoencoderKL(config)

    if pretrained is not None:
        logger.info(f"loading checkpoint from {pretrained}")
        msd = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(msd, strict=True)

    return model
