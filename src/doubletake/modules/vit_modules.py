import torch
import torch.nn as nn
import torch.nn.functional as F

from doubletake.modules.depth_anything_blocks import DPTHead
from doubletake.modules.layers import BasicBlock

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


class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            num_ch_cv=64,
            intermediate_layers=[2, 3, 4, 5],
            model_name='dinov2_vitb14',
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.intermediate_layers = intermediate_layers

        self.matching_project = nn.Linear(self.num_channels, 128)

        self.cv_block = nn.Sequential(
            BasicBlock(num_ch_cv, self.num_channels, stride=1),
            BasicBlock(self.num_channels, self.num_channels, stride=1),
        )

    def forward_matching_feats(self, cur_img, src_img):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, N, C, H, W = src_img.shape

        x = torch.cat([cur_img.unsqueeze(1), src_img], dim=1)
        x = x.view(B * (N+1), C, H, W)

        x = self.model.prepare_tokens_with_masks(x)

        # Last blocks are trained
        for blk in self.model.blocks[:1]:
            x = blk(x)

        matching_feats = self.matching_project(self.model.norm(x)[:, 1:])
        matching_feats = matching_feats.reshape((B, N+1, H // 14, W // 14, 128))
        matching_cur_feats = matching_feats[:, 0].permute(0, 3, 1, 2)
        matching_src_feats = matching_feats[:, 1:].permute(0, 1, 4, 2, 3)
        
        t = x[:, 0].reshape((B, N+1, self.num_channels))
        f = x[:, 1:].reshape((B, N+1, H // 14 * W // 14, self.num_channels))

        cur_token = t[:, 0]
        src_token = t[:, 1:]

        cur_feats = f[:, 0]
        src_feats = f[:, 1:]

        return (cur_token, cur_feats), (matching_cur_feats, matching_src_feats)
    
    
    def forward(self, cost_volume, cur_token, cur_feats):

        # Project cost volume
        B, C, H, W = cost_volume.shape
        x = self.cv_block(cost_volume)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((B, H * W, self.num_channels))
        
        cur_feats = cur_feats + x

        x = torch.cat([cur_token.unsqueeze(1), cur_feats], dim=1)

        feats = []
        for i, blk in enumerate(self.model.blocks[1:]):
            x = blk(x)
            if i+1 in self.intermediate_layers:
                feats.append(x)
        
        feats = [self.model.norm(out) for out in feats]
        return feats