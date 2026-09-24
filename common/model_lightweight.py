# Lightweight 2D-to-3D lifter baselines and a factory to build any of the
# repo's lifter models with a unified interface.

import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from common.model_cross import MixSTE2, Cross_Linformer, MixSTERELA, Block
from common.linearattention import LinearMultiheadAttention


class TemporalConvBlock(nn.Module):
    def __init__(self, width, dilation, dropout=0.1, kernel_size=3):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(width, width, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(width)
        self.norm2 = nn.BatchNorm1d(width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(torch.relu(self.norm1(self.conv1(x))))
        y = self.drop(torch.relu(self.norm2(self.conv2(y))))
        return x + y


class TemporalConvLifter(nn.Module):
    # VideoPose3D-style dilated temporal convolution lifter, but keeping all
    # frames (MixSTE-style full-sequence output): (b, f, n, 2) -> (b, f, n, 3)
    def __init__(self, num_joints=17, in_features=2, num_frames=243, width=512,
                 depth=4, dropout=0.1, out_features=3):
        super().__init__()
        self.num_joints = num_joints
        self.in_features = in_features
        self.out_features = out_features

        self.stem = nn.Conv1d(num_joints * in_features, width, 1)
        self.blocks = nn.ModuleList([
            TemporalConvBlock(width, dilation=3 ** i, dropout=dropout)
            for i in range(depth)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, num_joints * out_features, 1),
        )

    def forward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c -> b (n c) f')
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.head(x)
        x = rearrange(x, 'b (n c) f -> b f n c', n=n, c=self.out_features)
        return x


class MixSTELite(nn.Module):
    # MixSTE2-compatible forward contract ((b,f,n,2) -> (b,f,n,3)) built on
    # LinearMultiheadAttention (Linformer-style low-rank attention) to stay
    # lightweight.
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=128,
                 depth=4, num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        spatial_attention = partial(LinearMultiheadAttention, seq_len=num_joints)
        temporal_attention = partial(LinearMultiheadAttention, seq_len=num_frame)

        self.STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                attention=spatial_attention, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                attention=temporal_attention, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def STE_forward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TTE_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _ = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape) == 4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            x = self.STEblocks[i](x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            x = self.TTEblocks[i](x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        return x

    def forward(self, x, is_record=False):
        b, f, n, c = x.shape
        x = self.STE_forward(x)
        x = self.TTE_foward(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_foward(x)
        x = self.head(x)
        x = x.view(b, f, n, -1)
        return x


_LIGHTWEIGHT_PRESETS = {
    'temporal_conv': {
        'small': dict(width=256, depth=3),
        'base': dict(width=512, depth=4),
    },
    'mixste_lite': {
        'small': dict(embed_dim_ratio=64, depth=2, num_heads=4),
        'base': dict(embed_dim_ratio=128, depth=4, num_heads=8),
    },
    'linformer': {
        'small': dict(embed_dim_ratio=128, depth=2),
        'base': dict(embed_dim_ratio=256, depth=4),
    },
    'rela': {
        'small': dict(embed_dim_ratio=128, depth=2),
        'base': dict(embed_dim_ratio=256, depth=4),
    },
    'mixste': {
        'small': dict(embed_dim_ratio=256, depth=4),
        'base': dict(embed_dim_ratio=512, depth=8),
    },
}


def create_lifter(name, num_joints, num_frames, preset='base', **overrides):
    """Build a 2D->3D lifter by name.

    name: 'mixste' | 'mixste_lite' | 'linformer' | 'rela' | 'temporal_conv'
    preset: 'small' | 'base' (width/depth defaults for each model)
    overrides: hyperparameter overrides, e.g. embed_dim_ratio=512, depth=8,
               drop_path=0.1 (mapped to drop_path_rate), width=..., dropout=...
    """
    if preset not in ('small', 'base'):
        raise KeyError('Invalid preset: {}'.format(preset))
    params = dict(_LIGHTWEIGHT_PRESETS[name][preset]) if name in _LIGHTWEIGHT_PRESETS else {}
    drop_path = overrides.pop('drop_path', None)

    if name == 'mixste':
        params = dict(embed_dim_ratio=512, depth=8)
        params.update({k: v for k, v in overrides.items() if k in ('embed_dim_ratio', 'depth')})
        return MixSTE2(num_frame=num_frames, num_joints=num_joints, in_chans=2,
                       embed_dim_ratio=params['embed_dim_ratio'], depth=params['depth'],
                       num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                       drop_path_rate=drop_path if drop_path is not None else 0.1)

    if name == 'mixste_lite':
        params.update({k: v for k, v in overrides.items()
                       if k in ('embed_dim_ratio', 'depth', 'num_heads', 'mlp_ratio')})
        return MixSTELite(num_frame=num_frames, num_joints=num_joints, in_chans=2,
                          drop_path_rate=drop_path if drop_path is not None else 0.1,
                          **params)

    if name == 'linformer':
        params.update({k: v for k, v in overrides.items() if k in ('embed_dim_ratio', 'depth')})
        return Cross_Linformer(num_frame=num_frames, num_joints=num_joints, in_chans=2,
                               num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                               drop_path_rate=drop_path if drop_path is not None else 0.1,
                               **params)

    if name == 'rela':
        params.update({k: v for k, v in overrides.items() if k in ('embed_dim_ratio', 'depth')})
        return MixSTERELA(num_frame=num_frames, num_joints=num_joints, in_chans=2,
                          num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                          drop_path_rate=drop_path if drop_path is not None else 0.1,
                          **params)

    if name == 'temporal_conv':
        # accept embed_dim_ratio as an alias for width so generic callers work
        if 'embed_dim_ratio' in overrides and 'width' not in overrides:
            overrides['width'] = overrides['embed_dim_ratio']
        params.update({k: v for k, v in overrides.items() if k in ('width', 'depth', 'dropout')})
        return TemporalConvLifter(num_joints=num_joints, num_frames=num_frames, **params)

    raise KeyError('Invalid lifter name: {}'.format(name))
