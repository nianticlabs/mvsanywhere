import torch
import torch.nn as nn
import torch.nn.functional as F

from doubletake.modules.networks import double_basic_block


model_configs = {
    'dinov2_vits14': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'dinov2_vitb14': {'in_channels': 768, 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'dinov2_vitl14': {'in_channels': 1024, 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'dinov2_vitg14': {'in_channels': 1536, 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
    

class DPTHead(nn.Module):
    def __init__(
            self,
            model_name='dinov2_vits14',
            nclass=1,
            use_clstoken=False,
            use_bn=False
        ):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken

        in_channels = model_configs[model_name]['in_channels']
        features = model_configs[model_name]['features']
        out_channels = model_configs[model_name]['out_channels']
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        # self.cv_feat_fusers = nn.ModuleList([
        #     double_basic_block(out_channels[i] + cv_encoder_feat_channels[i], out_channels[i])
        #     for i in range(len(out_channels))
        # ])
        
        if self.use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
                    nn.Upsample(size=(int(420 // 2 ** i), int(560 // 2 ** i)), mode='bilinear', align_corners=True),
                    nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(head_features_2, nclass, kernel_size=1, stride=1, padding=0),
                    # nn.ReLU(True),
                    nn.Identity(),
                ) for i in range(4)
            ]
        )

        self.sky_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
                    nn.Upsample(size=(int(420 // 2 ** i), int(560 // 2 ** i)), mode='bilinear', align_corners=True),
                    nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(head_features_2, nclass, kernel_size=1, stride=1, padding=0),
                    # nn.ReLU(True),
                    nn.Identity(),
                ) for i in range(4)
            ]
        )

            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            # x = torch.cat([
            #     x,
            #     F.interpolate(cv_features[i], x.shape[2:], mode='bilinear')
            # ], dim=1)
            # x = self.cv_feat_fusers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        depth_outputs = {
            "log_depth_pred_s0_b1hw": self.output_convs[0](path_1),
            "log_depth_pred_s1_b1hw": self.output_convs[1](path_2),
            "log_depth_pred_s2_b1hw": self.output_convs[2](path_3),
            "log_depth_pred_s3_b1hw": self.output_convs[3](path_4),
            "sky_pred_s0_b1hw": self.output_convs[0](path_1),
            "sky_pred_s1_b1hw": self.output_convs[1](path_2),
            "sky_pred_s2_b1hw": self.output_convs[2](path_3),
            "sky_pred_s3_b1hw": self.output_convs[3](path_4)
        }
  
        return depth_outputs
    

    def load_da_weights(self, weights_path):
        da_state_dict = torch.load(weights_path)
        self.load_state_dict(
            {k.replace('depth_head.', ''): v for k, v in da_state_dict.items() if ('depth_head' in k) and ('output_conv' not in k)},
            strict=False
        )