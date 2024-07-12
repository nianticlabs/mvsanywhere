import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_elu=True, use_bn=False):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)
        if use_elu:
            self.non_lin = nn.ELU(inplace=True)
        else:
            self.non_lin = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.non_lin(x)

        x = self.conv2(x)
        x = self.non_lin(x)

        return x


class ConvUpsampleAndConcatBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_chns, use_elu=True, use_bn=False):
        super(ConvUpsampleAndConcatBlock, self).__init__()

        self.pre_concat_conv = ConvBlock(in_ch=in_ch, out_ch=out_ch, use_elu=use_elu, use_bn=use_bn)
        self.post_concat_conv = ConvBlock(
            in_ch=out_ch + skip_chns, out_ch=out_ch, use_elu=use_elu, use_bn=use_bn
        )

    def forward(self, x, cat_feats):
        x = self.pre_concat_conv(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, cat_feats], 1)
        x = self.post_concat_conv(x)

        return x


class SkipDecoder(nn.Module):
    def __init__(self, input_channels, use_bn=False):
        super(SkipDecoder, self).__init__()

        input_channels = input_channels[::-1]
        self.input_channels = input_channels
        self.output_channels = [256, 128, 64, 64]
        self.num_ch_dec = self.output_channels[::-1]

        self.block1 = ConvUpsampleAndConcatBlock(
            in_ch=input_channels[0],
            out_ch=self.output_channels[0],
            skip_chns=input_channels[1],
            use_bn=use_bn,
        )
        self.block2 = ConvUpsampleAndConcatBlock(
            in_ch=input_channels[1],
            out_ch=self.output_channels[1],
            skip_chns=self.input_channels[2],
            use_bn=use_bn,
        )
        self.block3 = ConvUpsampleAndConcatBlock(
            in_ch=input_channels[2],
            out_ch=self.output_channels[2],
            skip_chns=self.input_channels[3],
            use_bn=use_bn,
        )
        self.block4 = ConvUpsampleAndConcatBlock(
            in_ch=input_channels[3],
            out_ch=self.output_channels[3],
            skip_chns=self.input_channels[4],
            use_bn=use_bn,
        )

    def forward(self, features):
        output_features = {}
        x = features[-1]

        x = self.block1(x, features[-2])
        output_features[f"feature_s3_b1hw"] = x

        x = self.block2(x, features[-3])
        output_features[f"feature_s2_b1hw"] = x

        x = self.block3(x, features[-4])
        output_features[f"feature_s1_b1hw"] = x

        x = self.block4(x, features[-5])
        output_features[f"feature_s0_b1hw"] = x

        return output_features


class SkipDecoderRegression(SkipDecoder):
    def __init__(self, input_channels, use_bn=False):
        super(SkipDecoderRegression, self).__init__(input_channels, use_bn=use_bn)

        self.out1 = nn.Sequential(
            nn.Conv2d(self.output_channels[0], 128, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.out2 = nn.Sequential(
            nn.Conv2d(self.output_channels[1], 128, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.out3 = nn.Sequential(
            nn.Conv2d(self.output_channels[2], 128, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.out4 = nn.Sequential(
            nn.Conv2d(self.output_channels[3], 128, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, features):
        output_features = super(SkipDecoderRegression, self).forward(features)
        output_features[f"log_depth_pred_s3_b1hw"] = self.out1(output_features[f"feature_s3_b1hw"])
        output_features[f"log_depth_pred_s2_b1hw"] = self.out2(output_features[f"feature_s2_b1hw"])
        output_features[f"log_depth_pred_s1_b1hw"] = self.out3(output_features[f"feature_s1_b1hw"])
        output_features[f"log_depth_pred_s0_b1hw"] = self.out4(output_features[f"feature_s0_b1hw"])

        return output_features
