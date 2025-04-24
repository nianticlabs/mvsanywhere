from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from mvsanywhere.modules.depth_anything_blocks import DPTHead
from mvsanywhere.modules.layers import BasicBlock
from mvsanywhere.modules.networks import double_basic_block


DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PytorchMemEffAttention(Attention):
    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p).transpose(1, 2)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def use_memeffattn_in_model(model):

    for i in range(len(model.blocks)):

        slow_attn = model.blocks[i].attn

        meff_attn = PytorchMemEffAttention(
            dim=slow_attn.qkv.in_features,
            num_heads=slow_attn.num_heads,
        )
        meff_attn.qkv = slow_attn.qkv
        meff_attn.attn_drop = slow_attn.attn_drop
        meff_attn.proj = slow_attn.proj
        meff_attn.proj_drop = slow_attn.proj_drop

        model.blocks[i].attn = meff_attn

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
        use_memeffattn_in_model(self.model)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_intermediate_layers = len(self.model.blocks) if num_intermediate_layers == -1 else num_intermediate_layers

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 16, W // 16].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """
        scale_factor = 14 / 16 
        x = F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        return self.model.get_intermediate_layers(x, self.num_intermediate_layers, return_class_token=True)

    def load_da_weights(self, weights_path):
        try:
            da_state_dict = torch.load(weights_path)
            self.load_state_dict(
                {k.replace('pretrained', 'model'): v for k, v in da_state_dict.items() if 'pretrained' in k},
            )
        except:
            logger.info("Couldn't load DA weights from {weights_path}.")


class CostVolumePatchEmbed(nn.Module):

    def __init__(
        self,
        num_ch_cv,
        num_feats,
        num_ch_outs = [128, 256],
        num_ch_proj = [96, 192],
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

            if i < 2:
                self.convs[f"conv_{i}"] = nn.Sequential(
                    BasicBlock(num_ch_proj[i] + num_ch_out, num_ch_out, stride=1),
                    BasicBlock(num_ch_out, num_ch_out, stride=1),
                )

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=num_feats,
                out_channels=chns,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for chns in num_ch_proj
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=num_ch_proj[0],
                out_channels=num_ch_proj[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=num_ch_proj[1],
                out_channels=num_ch_proj[1],
                kernel_size=2,
                stride=2,
                padding=0),
        ])
        
    def forward(self, x, img_feats):
        # resize such that 2 downsamples will give 1/14th resolution 
        B, C, H, W = x.shape

        for i in range(3):


            x = self.convs[f"ds_conv_{i}"](x)

            if i < 2:
                img_feat = img_feats[i][0].reshape((B, H * 4 // 16, W * 4 // 16, self.num_feats))
                img_feat = img_feat.permute(0, 3, 1, 2)
                img_feat = self.projects[i](img_feat)
                img_feat = self.resize_layers[i](img_feat)
                x = torch.cat([x, img_feat], dim=1)
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
            feat_fuser_layers_idx=[2, 5, 8, 11],
            intermediate_layers_idx=[2, 5, 8, 11]
    ):
        super().__init__()
        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_ch_cv = num_ch_cv

        # Load DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        use_memeffattn_in_model(self.model)

        self.feat_fuser_layers_idx = feat_fuser_layers_idx
        self.intermediate_layers_idx = intermediate_layers_idx

        self.cv_feat_fusers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.num_channels, self.num_channels),
                nn.ReLU(True),
            )
            for i in range(len(feat_fuser_layers_idx))
        ])

        self.patch_embed = CostVolumePatchEmbed(
            num_ch_cv,
            self.num_channels,
        )

    def forward(self, x, img_feats):
        cv_embed_layers = img_feats[:2]
        fuser_layers = img_feats[2:]

        x = self.prepare_tokens_with_masks(x, cv_embed_layers)

        feats = []
        for i, blk in enumerate(self.model.blocks):

            # Fuse with mono branch ViT layer
            if i in self.feat_fuser_layers_idx:
                fuse_layer_i = self.feat_fuser_layers_idx.index(i)
                fuse_layer = fuser_layers[fuse_layer_i]
                x = x + self.cv_feat_fusers[fuse_layer_i](torch.cat([fuse_layer[1].unsqueeze(1), fuse_layer[0]], dim=1))
            # Run CV branch ViT block
            x = blk(x)

            # Save intermediate feat
            if i in self.intermediate_layers_idx:
                feats.append((x[:, 1:], x[:, 0]))
                
        return feats

    def prepare_tokens_with_masks(self, x, img_features, masks=None):
        B, C, H, W = x.shape
        x = self.patch_embed(x, img_features)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.model.interpolate_pos_encoding(x, H * 4 * 14 / 16, W * 4 * 14 / 16)

        return x