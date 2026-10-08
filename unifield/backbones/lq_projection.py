from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time


CACHE_T = 2


# =============================================================================
# 双向 3D 卷积（非因果）- 用于 MRI 3D 数据
# =============================================================================
class BidirectionalConv3d(nn.Conv3d):
    """
    Bidirectional 3D convolution for MRI 3D data.
    Unlike CausalConv3d, this uses symmetric padding in temporal dimension,
    allowing the model to see both past and future slices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 对称填充：时间维度两边都填充
        self._padding = (self.padding[2], self.padding[2],  # W 方向
                         self.padding[1], self.padding[1],  # H 方向
                         self.padding[0], self.padding[0])  # F 方向（双向对称）
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        # 忽略 cache_x，对于 MRI 3D 不需要缓存
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

class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            # print(cache_x.shape, x.shape)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
            # print('cache!')
        x = F.pad(x, padding, mode='replicate') # mode='replicate'
        # print(x[0,0,:,0,0])

        return super().forward(x)

class PixelShuffle3d(nn.Module):
    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x):
        # x: (B, C, F, H, W)
        return rearrange(x,
                         'b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w',
                         ff=self.ff, hh=self.hh, ww=self.ww)

class Buffer_LQ4x_Proj(nn.Module):

    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        self.conv1 = CausalConv3d(in_dim*self.ff*self.hh*self.ww, self.hidden_dim1, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])

        self.clip_idx = 0
        self.cache = {'conv1': None, 'conv2': None}  # Initialize cache here

    def forward(self, video):
        self.clear_cache()
        # x: (B, C, F, H, W)

        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)
        # print(video.shape)

        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:,:,i*4:(i+1)*4,:,:])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            if i == 0:
                continue
            x = self.conv2(x, self.cache['conv2'])
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x = torch.cat(out_x, dim = 2)
        # print(out_x.shape)
        out_x = rearrange(out_x, 'b c f h w -> b (f h w) c')
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))
        return outputs

    def clear_cache(self):
        self.cache = {}
        self.cache['conv1'] = None
        self.cache['conv2'] = None
        self.clip_idx = 0

    def stream_forward(self, video_clip):
        '''
        if self.clip_idx == 0:
            # self.clear_cache()
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            self.clip_idx += 1
            return None
        else:
        '''
        x = self.pixel_shuffle(video_clip)
        cache1_x = x[:, :, -CACHE_T:, :, :].clone()
        self.cache['conv1'] = cache1_x
        x = self.conv1(x, self.cache['conv1'])
        x = self.norm1(x)
        x = self.act1(x)
        cache2_x = x[:, :, -CACHE_T:, :, :].clone()
        self.cache['conv2'] = cache2_x
        x = self.conv2(x, self.cache['conv2'])
        x = self.norm2(x)
        x = self.act2(x)
        out_x = rearrange(x, 'b c f h w -> b (f h w) c')
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))
        self.clip_idx += 1
        return outputs

class Causal_LQ4x_Proj(nn.Module):

    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        self.conv1 = CausalConv3d(in_dim*self.ff*self.hh*self.ww, self.hidden_dim1, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])

        self.clip_idx = 0

    def forward(self, video):
        self.clear_cache()
        # x: (B, C, F, H, W)

        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)
        # print(video.shape)

        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:,:,i*4:(i+1)*4,:,:])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            if i == 0:
                self.cache['conv2'] = cache2_x
                continue
            x = self.conv2(x, self.cache['conv2'])
            self.cache['conv2'] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x = torch.cat(out_x, dim = 2)
        out_x = rearrange(out_x, 'b c f h w -> b (f h w) c')
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))
        return outputs

    def clear_cache(self):
        self.cache = {}
        self.cache['conv1'] = None
        self.cache['conv2'] = None
        self.clip_idx = 0

    def stream_forward(self, video_clip):
        if self.clip_idx == 0:
            # self.clear_cache()
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv2(x, self.cache['conv2'])
            self.cache['conv2'] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, 'b c f h w -> b (f h w) c')
            outputs = []
            for i in range(self.layer_num):
                outputs.append(self.linear_layers[i](out_x))
            self.clip_idx += 1
            return outputs


# =============================================================================
# 双向 LQ Proj-In（非因果）- 专为 MRI 3D 数据设计
# =============================================================================
class Bidirectional_LQ4x_Proj(nn.Module):
    """
    Bidirectional LQ Projection for MRI 3D data.

    Unlike Buffer_LQ4x_Proj and Causal_LQ4x_Proj which use causal convolutions
    (only seeing past frames), this module uses bidirectional convolutions
    that can see both past and future slices.

    This is appropriate for MRI 3D data where all slices exist simultaneously
    and there is no temporal causality constraint.
    """

    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        # 使用双向卷积替代因果卷积
        self.conv1 = BidirectionalConv3d(
            in_dim * self.ff * self.hh * self.ww,
            self.hidden_dim1,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1)
        )
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = BidirectionalConv3d(
            self.hidden_dim1,
            self.hidden_dim2,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1)
        )
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.linear_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)
        ])

        self.clip_idx = 0
        self.cache = {'conv1': None, 'conv2': None}

    def forward(self, video):
        """
        完整视频的前向传播（双向处理）

        与因果版本 Buffer_LQ4x_Proj 保持相同的填充和分块处理逻辑，
        确保输出帧数一致。唯一区别是使用双向卷积。

        Args:
            video: (B, C, F, H, W) - 完整的 MRI 3D 数据

        Returns:
            outputs: list of tensors, 每层一个输出
        """
        # 与因果版本完全相同的帧数处理逻辑
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4

        # 与因果版本相同：前面填充 3 帧（首帧重复）
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)

        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:, :, i*4:(i+1)*4, :, :])

            # 双向卷积不需要 cache，直接前向
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.act1(x)

            # 第一个迭代不输出（与因果版本保持一致）
            if i == 0:
                continue

            x = self.conv2(x)
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)

        out_x = torch.cat(out_x, dim=2)
        out_x = rearrange(out_x, 'b c f h w -> b (f h w) c')

        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))

        return outputs

    def clear_cache(self):
        """清理缓存（保持接口兼容性）"""
        self.cache = {'conv1': None, 'conv2': None}
        self.clip_idx = 0

    def stream_forward(self, video_clip):
        """
        流式前向传播 - 与 Buffer_LQ4x_Proj 保持一致，不漏帧

        Args:
            video_clip: (B, C, F, H, W) - 视频片段

        Returns:
            outputs: list of tensors, 每层一个输出
        """
        x = self.pixel_shuffle(video_clip)
        cache1_x = x[:, :, -CACHE_T:, :, :].clone()
        self.cache['conv1'] = cache1_x
        x = self.conv1(x, self.cache['conv1'])
        x = self.norm1(x)
        x = self.act1(x)
        cache2_x = x[:, :, -CACHE_T:, :, :].clone()
        self.cache['conv2'] = cache2_x
        x = self.conv2(x, self.cache['conv2'])
        x = self.norm2(x)
        x = self.act2(x)
        out_x = rearrange(x, 'b c f h w -> b (f h w) c')
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))
        self.clip_idx += 1
        return outputs
