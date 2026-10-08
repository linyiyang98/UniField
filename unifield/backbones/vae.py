"""
Bidirectional (Non-Causal) VAE for MRI 3D data.

This module provides bidirectional versions of the WanVideoVAE components.
Instead of causal padding (only seeing past frames), these use symmetric padding
to see both past and future slices - appropriate for MRI 3D volumes.

The weights are compatible with the original causal VAE, so pretrained weights
can be loaded directly.
"""

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def check_is_instance(model, module_class):
    if isinstance(model, module_class):
        return True
    if hasattr(model, "module") and isinstance(model.module, module_class):
        return True
    return False


class BidirectionalConv3d(nn.Conv3d):
    """
    Bidirectional 3D convolution for MRI 3D data.
    Unlike CausalConv3d, this uses symmetric padding in temporal dimension,
    allowing the model to see both past and future slices.

    The weights are identical to CausalConv3d, only padding differs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 对称填充：时间维度两边都填充相同的量
        # 原始 CausalConv3d: (W_left, W_right, H_top, H_bottom, T_front, T_back)
        #                  = (padding[2], padding[2], padding[1], padding[1], 2*padding[0], 0)
        # 双向版本: 时间维度改为对称填充
        self._padding = (self.padding[2], self.padding[2],  # W 方向
                         self.padding[1], self.padding[1],  # H 方向
                         self.padding[0], self.padding[0])  # F 方向（双向对称）
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        # 忽略 cache_x，对于 MRI 3D 不需要缓存（非流式处理）
        x = F.pad(x, self._padding, mode='replicate')
        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class BidirectionalResample(nn.Module):
    """
    Bidirectional version of Resample module.
    Uses symmetric temporal padding instead of causal padding.
    """

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = BidirectionalConv3d(dim,
                                                  dim * 2, (3, 1, 1),
                                                  padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = BidirectionalConv3d(dim,
                                                  dim, (3, 1, 1),
                                                  stride=(2, 1, 1),
                                                  padding=(1, 0, 0))  # 注意：stride=2 时需要调整

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()

        if self.mode == 'upsample3d':
            # 时间上采样：使用双向卷积
            x = self.time_conv(x)
            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
            x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            # 时间下采样：使用双向卷积
            x = self.time_conv(x)

        return x


class BidirectionalResidualBlock(nn.Module):
    """
    Bidirectional version of ResidualBlock.
    Uses BidirectionalConv3d instead of CausalConv3d.
    """

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            BidirectionalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            BidirectionalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = BidirectionalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Self-attention with a single head.
    Note: This is already non-causal (no temporal masking).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(
            0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention (no causal mask)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class BidirectionalEncoder3d(nn.Module):
    """
    Bidirectional version of Encoder3d.
    Uses symmetric temporal padding for MRI 3D data.
    """

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = BidirectionalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(BidirectionalResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(BidirectionalResample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            BidirectionalResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            BidirectionalResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            BidirectionalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            x = layer(x)

        ## middle
        for layer in self.middle:
            x = layer(x)

        ## head
        for layer in self.head:
            x = layer(x)
        return x


class BidirectionalDecoder3d(nn.Module):
    """
    Bidirectional version of Decoder3d.
    Uses symmetric temporal padding for MRI 3D data.
    """

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = BidirectionalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            BidirectionalResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            BidirectionalResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(BidirectionalResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(BidirectionalResample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            BidirectionalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        x = self.conv1(x)

        ## middle
        for layer in self.middle:
            x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            x = layer(x)

        ## head
        for layer in self.head:
            x = layer(x)
        return x


class BidirectionalVideoVAE_(nn.Module):
    """
    Bidirectional version of VideoVAE_.
    Processes the entire volume at once without causal constraints.
    """

    def __init__(self,
                 dim=96,
                 z_dim=16,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = BidirectionalEncoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                               attn_scales, self.temperal_downsample, dropout)
        self.conv1 = BidirectionalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = BidirectionalConv3d(z_dim, z_dim, 1)
        self.decoder = BidirectionalDecoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                               attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        """
        Encode volume in chunks (same as causal VAE but without cache).

        分块方式与因果 VAE 完全一致：第1帧单独处理，之后每4帧一组。
        双向版本不需要 cache，因为使用对称 padding。

        Args:
            x: Input video (B, C, T, H, W)
            scale: Normalization scale
        """
        # MRI is a spatial volume, so symmetric convolutions must see the
        # complete sampled depth. The legacy video loop consumed 1+4*k slices
        # and silently omitted the final three slices for an 8n-depth volume.
        out = self.encoder(x)

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        return mu

    def decode(self, z, scale):
        """
        Decode latents in chunks (same as causal VAE but without cache).

        分块方式与因果 VAE 完全一致：逐帧处理 latent。
        双向版本不需要 cache，因为使用对称 padding。

        Args:
            z: Latent tensor (B, C, T, H, W)
            scale: Normalization scale
        """
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]

        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            if i == 0:
                out = self.decoder(x[:, :, i:i + 1, :, :])
            else:
                out_ = self.decoder(x[:, :, i:i + 1, :, :])
                out = torch.cat([out, out_], 2)
        return out

    def stream_decode(self, z, scale):
        """
        For compatibility - same as decode for bidirectional version.
        """
        return self.decode(z, scale)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        """For compatibility with causal version - no-op for bidirectional."""
        pass


class BidirectionalWanVideoVAE(nn.Module):
    """
    Bidirectional version of WanVideoVAE for MRI 3D data.

    Can load weights from the original causal WanVideoVAE directly,
    since only the padding method differs.
    """

    def __init__(self, z_dim=16, dim=96):
        super().__init__()

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = BidirectionalVideoVAE_(z_dim=z_dim, dim=dim).eval().requires_grad_(False)
        self.upsampling_factor = 8

    def single_encode(self, x, device):
        """Encode a single video without tiling."""
        x = x.to(device)
        return self.model.encode(x, self.scale).to("cpu")

    def single_decode(self, z, device):
        """Decode a single latent without tiling."""
        z = z.to(device)
        video = self.model.decode(z, self.scale)
        return self.postprocess(video).to("cpu")

    def postprocess(self, video):
        """Post-process decoded video."""
        return video.clamp_(-1, 1)

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """
        Encode videos to latent space.

        API compatible with original WanVideoVAE.encode().
        """
        videos = [video.to("cpu") for video in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size_scaled = (tile_size[0] * 8, tile_size[1] * 8)
                tile_stride_scaled = (tile_stride[0] * 8, tile_stride[1] * 8)
                hidden_state = self.tiled_encode(video, device, tile_size_scaled, tile_stride_scaled)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states

    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """
        Decode latents to videos.

        API compatible with original WanVideoVAE.decode().
        """
        hidden_states = [hidden_state.to("cpu") for hidden_state in hidden_states]
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride)
            else:
                video = self.single_decode(hidden_state, device)
            video = video.squeeze(0)
            videos.append(video)
        videos = torch.stack(videos)
        return videos

    def stream_decode(self, hidden_states, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """For compatibility with causal version - uses regular decode."""
        hidden_states = [hidden_state for hidden_state in hidden_states]
        assert len(hidden_states) == 1
        hidden_state = hidden_states[0]
        video = self.model.decode(hidden_state.unsqueeze(0), self.scale).squeeze(0)
        return video

    def clear_cache(self):
        """For compatibility - no-op for bidirectional."""
        self.model.clear_cache()

    def tiled_encode(self, video, device, tile_size, tile_stride):
        """Tiled encoding for large videos."""
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h - stride_h >= 0 and h - stride_h + size_h >= H):
                continue
            for w in range(0, W, stride_w):
                if (w - stride_w >= 0 and w - stride_w + size_w >= W):
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        # 计算输出尺寸 (空间维度 /8, 时间维度根据downsample)
        out_T = (T + 3) // 4  # 时间下采样
        out_H = H // self.upsampling_factor
        out_W = W // self.upsampling_factor

        # 使用 z_dim=16
        weight = torch.zeros((1, 1, out_T, out_H, out_W), dtype=video.dtype, device=data_device)
        values = torch.zeros((1, 16, out_T, out_H, out_W), dtype=video.dtype, device=data_device)

        for h, h_, w, w_ in tasks:
            video_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_batch = self.model.encode(video_batch, self.scale).to(data_device)

            out_h = h // self.upsampling_factor
            out_h_ = min(h_ // self.upsampling_factor, out_H)
            out_w = w // self.upsampling_factor
            out_w_ = min(w_ // self.upsampling_factor, out_W)

            mask = self.build_mask(
                hidden_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) // self.upsampling_factor,
                    (size_w - stride_w) // self.upsampling_factor
                )
            ).to(dtype=video.dtype, device=data_device)

            target_h = slice(out_h, out_h_)
            target_w = slice(out_w, out_w_)
            source_h = slice(0, out_h_ - out_h)
            source_w = slice(0, out_w_ - out_w)

            weight[:, :, :, target_h, target_w] += mask[:, :, :, source_h, source_w]
            values[:, :, :, target_h, target_w] += hidden_batch[:, :, :, source_h, source_w] * mask[:, :, :, source_h, source_w]

        values = values / weight
        return values.to(device)

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        # 对于双向 VAE，时间维度的输出与输入相同比例
        out_T = T * 4 - 3
        weight = torch.zeros((1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
                            dtype=hidden_states.dtype, device=data_device)
        values = torch.zeros((1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
                            dtype=hidden_states.dtype, device=data_device)

        for h, h_, w, w_ in tasks:
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=(
                    (size_h - stride_h) * self.upsampling_factor,
                    (size_w - stride_w) * self.upsampling_factor
                )
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = slice(h * self.upsampling_factor, min(h_ * self.upsampling_factor, H * self.upsampling_factor))
            target_w = slice(w * self.upsampling_factor, min(w_ * self.upsampling_factor, W * self.upsampling_factor))

            source_h = slice(0, target_h.stop - target_h.start)
            source_w = slice(0, target_w.stop - target_w.start)

            weight[:, :, :, target_h, target_w] += mask[:, :, :, source_h, source_w]
            values[:, :, :, target_h, target_w] += hidden_states_batch[:, :, :, source_h, source_w] * mask[:, :, :, source_h, source_w]

        values = values / weight
        return values.to(device)


def load_causal_weights_to_bidirectional(bidirectional_vae, causal_state_dict):
    """
    Load weights from a causal WanVideoVAE to a BidirectionalWanVideoVAE.

    The weights are compatible because only the padding method differs.

    Args:
        bidirectional_vae: BidirectionalWanVideoVAE instance
        causal_state_dict: state_dict from causal WanVideoVAE

    Returns:
        missing_keys, unexpected_keys
    """
    # 权重键名应该是相同的，只是模块类型不同
    return bidirectional_vae.load_state_dict(causal_state_dict, strict=False)
