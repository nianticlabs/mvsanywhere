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


class CNNCVEncoder(nn.Module):
    def __init__(
            self, 
            model_name,
            num_ch_cv,
            num_ch_outs
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        model_configs = MODEL_CONFIGS[model_name]

        self.convs = nn.ModuleDict()
        self.num_ch_enc = []

        self.num_blocks = len(num_ch_outs)

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=model_configs["in_channels"],
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in model_configs["out_channels"]
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=model_configs["out_channels"][0],
                out_channels=model_configs["out_channels"][0],
                kernel_size=8,
                stride=8,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=model_configs["out_channels"][1],
                out_channels=model_configs["out_channels"][1],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=model_configs["out_channels"][2],
                out_channels=model_configs["out_channels"][2],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=model_configs["out_channels"][4],
                out_channels=model_configs["out_channels"][4],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        for i in range(self.num_blocks):
            num_ch_in = num_ch_cv if i == 0 else num_ch_outs[i - 1]
            num_ch_out = num_ch_outs[i]
            self.convs[f"ds_conv_{i}"] = BasicBlock(
                num_ch_in, num_ch_out, stride=1 if i == 0 else 2
            )

            self.convs[f"conv_{i}"] = nn.Sequential(
                BasicBlock(model_configs["out_channels"][i+1] + num_ch_out, num_ch_out, stride=1),
                BasicBlock(num_ch_out, num_ch_out, stride=1),
            )
            self.num_ch_enc.append(num_ch_out)

    def forward(self, x, img_feats):

        # Reshape feat and project
        f = img_feats[0][0]
        f = f.permute(0, 2, 1).reshape((f.shape[0], f.shape[-1], 24, 32))
        f = self.projects[0](f)
        f = self.resize_layers[0](f)

        outputs = [f]
        x = F.interpolate(x, [96, 128], mode="nearest")
        for i in range(self.num_blocks):
            x = self.convs[f"ds_conv_{i}"](x)
            
            # Reshape feat and project
            f = img_feats[i + 1][0]
            f = f.permute(0, 2, 1).reshape((f.shape[0], f.shape[-1], 24, 32))
            f = self.projects[i + 1](f)
            f = self.resize_layers[i + 1](f)

            x = torch.cat([x, f], dim=1)
            x = self.convs[f"conv_{i}"](x)
            outputs.append(x)
        return outputs

    def load_da_weights(self, weights_path):
        da_state_dict = torch.load(weights_path)
        for i in range(4):
            self.projects[i+1].load_state_dict(
                {k.replace(f'depth_head.projects.{i}.', ''): v for k, v in da_state_dict.items() if f'depth_head.projects.{i}' in k},
            )
            self.resize_layers[i+1].load_state_dict(
                {k.replace(f'depth_head.resize_layers.{i}.', ''): v for k, v in da_state_dict.items() if f'depth_head.resize_layers.{i}' in k},
            )


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

        scale_factor = 16 / 14 
        x = F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

        for i in range(3):


            x = self.convs[f"ds_conv_{i}"](x)

            if i < 2:
                img_feat = img_feats[i][0].reshape((B, H * 4 // 14, W * 4 // 14, self.num_feats))
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
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

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
        # TODO: make this work

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
        B, nc, w, h = x.shape
        x = self.patch_embed(x, img_features)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.model.interpolate_pos_encoding(x, w * 4, h * 4)

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