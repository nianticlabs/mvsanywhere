from functools import partial
from itertools import repeat
import collections.abc

import torch
import torch.nn as nn
import torch.nn.functional as F

from doubletake.modules.depth_anything_blocks import DPTHead
from doubletake.modules.layers import BasicBlock
from doubletake.modules.networks import double_basic_block

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

MODEL_CONFIGS = {
    'dinov2_vits14': {'in_channels': 384, 'features': 64, 'out_channels': [32, 48, 96, 192, 384]},
    'dinov2_vitb14': {'in_channels': 768, 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'dinov2_vitl14': {'in_channels': 1024, 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'dinov2_vitg14': {'in_channels': 1536, 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

DINOV2_NUM_BLOCKS = {
    'dinov2_vits14': 12,
    'dinov2_vitb14': 12,
    'dinov2_vitl14': 24,
}

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)
               
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, query, key, value):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        
        q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, 2, self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4)
        v = self.projk(value).reshape(B, Nv, 2, self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4)

        attn = (q.unsqueeze(-2) @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).squeeze(-2).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        y_ = self.norm_y(y)
        x_ = self.norm2(x)

        # Cross attention to self patch and same mono patch
        kv = torch.stack([x_, y_], dim=-2)
        x = x + self.drop_path(self.cross_attn(x_, kv, kv))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y


class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_intermediate_layers=-1,
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_intermediate_layers = len(self.model.blocks) if num_intermediate_layers == -1 else num_intermediate_layers


    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """
        x = F.interpolate(x, scale_factor=14/16, mode='bilinear')
        return self.model.get_intermediate_layers(x, self.num_intermediate_layers, return_class_token=True)


    def load_da_weights(self, weights_path):
        da_state_dict = torch.load(weights_path)
        self.load_state_dict(
            {k.replace('pretrained', 'model'): v for k, v in da_state_dict.items() if 'pretrained' in k},
        )


class DepthAnything(nn.Module):

    def __init__(
            self,
            cv_encoder_feat_channel,
            model_name="dinov2_vitb14",
            intermediate_layers=4,
    ):
        super().__init__()
        self.dinov2 = DINOv2(model_name=model_name, num_intermediate_layers=intermediate_layers)
        self.depth_head = DPTHead(cv_encoder_feat_channel, model_name=model_name, nclass=1)

    def forward(self, img, cv_feats):
        h, w = img.shape[-2:]
        vit_feats = self.dinov2(img)

        patch_h, patch_w = h // 14, w // 14
        return self.depth_head(cv_feats, vit_feats, patch_h, patch_w)
    
    def load_da_weights(self, weights_path):
        self.dinov2.load_da_weights(weights_path)
        self.depth_head.load_da_weights(weights_path)


class CostVolumePatchEmbed(nn.Module):

    def __init__(
        self,
        num_ch_cv,
        num_feats,
        num_ch_outs = [128, 256],
        num_ch_proj = [16],
        patch_size = [14, 14],
    ):

        super().__init__()
        self.num_ch_outs = num_ch_outs
        self.num_ch_cv = num_ch_cv
        self.patch_size = patch_size
        self.num_feats = num_feats
        self.convs = nn.ModuleDict()

        for i in range(3):
            num_ch_in = num_ch_cv if i == 0 else num_ch_outs[i - 1]
            num_ch_out = (num_ch_outs + [self.num_feats])[i]
            self.convs[f"ds_conv_{i}"] = BasicBlock(
                num_ch_in, num_ch_out, stride=1 if i == 0 else 2
            )

            self.convs[f"conv_{i}"] = nn.Sequential(
                BasicBlock(num_ch_out, num_ch_out, stride=1),
                BasicBlock(num_ch_out, num_ch_out, stride=1),
            )
        
    def forward(self, x):
        # resize such that 2 downsamples will give 1/14th resolution 
        B, C, H, W = x.shape

        for i in range(3):
            x = self.convs[f"ds_conv_{i}"](x)
            x = self.convs[f"conv_{i}"](x)

        x = self.patch_embed(x)  # B HW C
        return x 

    def patch_embed(self, x):
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B HW C
        return x


class ViTCVEncoder(nn.Module):

    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_ch_cv=64,
            intermediate_layers_idx=[2, 5, 8, 11]
    ):
        super().__init__()
        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_ch_cv = num_ch_cv
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False)

        self.intermediate_layers_idx = intermediate_layers_idx

        self.model.blocks = nn.ModuleList([
            CrossBlock(self.num_channels, 12, 4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for i in range(DINOV2_NUM_BLOCKS[model_name])
        ])

        self.patch_embed = CostVolumePatchEmbed(
            num_ch_cv,
            self.num_channels,
        )

    def forward(self, cv, img_features):
        # TODO: make this work

        # Cost volume branch
        x = self.prepare_tokens_with_masks(cv)

        feats = []
        for i, blk in enumerate(self.model.blocks):

            # Cross attention between mono feats and cv feats
            x, _ = blk(x, torch.cat([img_features[i][1].unsqueeze(1), img_features[i][0]], dim=1))

            # Save intermediate feat
            if i in self.intermediate_layers_idx:
                feats.append((x[:, 1:], x[:, 0]))
                
        return feats

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.model.interpolate_pos_encoding(x, w * 4 * 14 / 16, h * 4 * 14 / 16)

        if self.model.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.model.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def load_da_weights(self, weights_path):
        da_state_dict = torch.load(weights_path)
        self.load_state_dict(
            {k.replace('pretrained', 'model'): v for k, v in da_state_dict.items() if 'pretrained' in k},
            strict=False
        )