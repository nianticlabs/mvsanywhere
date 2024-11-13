import math

import torch
import torch.nn as nn
import numpy as np
from torch.hub import load_state_dict_from_url

from .registry import register_model
from .helpers import build_model_with_cfg
from .blocks.dispnet_context_encoder import DispnetContextEncoder
from .blocks.dispnet_encoder import DispnetEncoder
from .blocks.planesweep_corr import PlanesweepCorrelation
from .blocks.learned_fusion import LearnedFusion
from .blocks.dispnet_costvolume_encoder import DispnetCostvolumeEncoder
from .blocks.dispnet_decoder import DispnetDecoder

from rmvd.utils import get_torch_model_device, to_numpy, to_torch, select_by_index, exclude_index
from rmvd.data.transforms import ResizeInputs


class RobustMVD(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = DispnetEncoder()
        self.context_encoder = DispnetContextEncoder()
        self.corr_block = PlanesweepCorrelation()
        self.fusion_block = LearnedFusion()
        self.fusion_enc_block = DispnetCostvolumeEncoder()
        self.decoder = DispnetDecoder()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv3d) or isinstance(
                    m, nn.ConvTranspose3d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images, poses, intrinsics, keyview_idx, **_):

        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)

        intrinsics_key = select_by_index(intrinsics, keyview_idx)
        intrinsics_source = exclude_index(intrinsics, keyview_idx)

        source_to_key_transforms = exclude_index(poses, keyview_idx)

        all_enc_key, enc_key = self.encoder(image_key)
        enc_sources = [self.encoder(image_source)[1] for image_source in images_source]

        ctx = self.context_encoder(enc_key)

        corrs, masks = self.corr_block(feat_key=enc_key, intrinsics_key=intrinsics_key, feat_sources=enc_sources,
                                       source_to_key_transforms=source_to_key_transforms,
                                       intrinsics_sources=intrinsics_source)

        fused_corr, fused_mask = self.fusion_block(corrs=corrs, masks=masks)

        all_enc_fused, enc_fused = self.fusion_enc_block(corr=fused_corr, ctx=ctx)

        dec = self.decoder(enc_fused=enc_fused, all_enc={**all_enc_key, **all_enc_fused})

        pred = {
            'depth': 1 / (dec['invdepth'] + 1e-9),
            'depth_uncertainty': torch.exp(dec['invdepth_log_b']) / (dec['invdepth'] + 1e-9)
        }
        aux = dec

        return pred, aux

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        orig_ht, orig_wd = images[0].shape[-2:]
        ht, wd = int(math.ceil(orig_ht / 64.0) * 64.0), int(math.ceil(orig_wd / 64.0) * 64.0)
        if (orig_ht != ht) or (orig_wd != wd):
            resized = ResizeInputs(size=(ht, wd))({'images': images, 'intrinsics': intrinsics})
            images = resized['images']
            intrinsics = resized['intrinsics']

        # normalize images
        images = [image / 255.0 - 0.4 for image in images]

        # model works with relative intrinsics:
        scale_arr = np.array([[wd]*3, [ht]*3, [1.]*3], dtype=np.float32)  # 3, 3
        intrinsics = [intrinsic / scale_arr for intrinsic in intrinsics]

        images, keyview_idx, poses, intrinsics, depth_range = \
            to_torch((images, keyview_idx, poses, intrinsics, depth_range), device=device)

        sample = {
            'images': images,
            'keyview_idx': keyview_idx,
            'poses': poses,
            'intrinsics': intrinsics,
            'depth_range': depth_range,
        }
        return sample

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


class RobustMVD_WrappedForMeshing(RobustMVD):

    def __init__(self):
        super().__init__()
        pretrained_weights = 'https://lmb.informatik.uni-freiburg.de/people/schroepp/weights/robustmvd.pt'

        state_dict = load_state_dict_from_url(
            pretrained_weights, map_location='cpu',
            progress=True,
            check_hash=True)

        self.load_state_dict(state_dict, strict=True)
        self.eval()
        self = self.cuda()

    def forward(
        self,
        phase: str,
        cur_data: dict,
        src_data: dict,
        unbatched_matching_encoder_forward: bool,
        return_mask:bool,
        raw_mast3r_pred: bool = False,
    ):
        batch_size = cur_data['image_b3hw'].shape[0]

        pred_depths = []

        # do each item in the batch separately for ease
        for batch_idx in range(batch_size):
            images = torch.vstack((
                cur_data['image_b3hw'][batch_idx][None, ...],
                src_data['image_b3hw'][batch_idx]
            ))

            # Take 'full' scale intrinsics
            intrinsics = torch.vstack((
                cur_data['K_s0_b44'][batch_idx][None, ...],
                src_data['K_s0_b44'][batch_idx],
            ))

            # Do all poses relative to the cur frame
            poses = [torch.eye(4).cuda()]
            cur_frame_pose = cur_data['world_T_cam_b44'][batch_idx]
            for _pose in src_data['cam_T_world_b44'][batch_idx]:
                poses.append(torch.inverse(_pose @ cur_frame_pose))
            poses = torch.stack(poses)

            pred, _ = super().forward(
                images=images.unsqueeze(1),
                intrinsics=intrinsics.unsqueeze(1),
                poses=poses.unsqueeze(1),
                min_depth=None,
                max_depth=None,
                previous_shape=None,
                keyview_idx=0,
            )

            pred_depths.append(pred['depth'])

        pred_depth = torch.vstack(pred_depths)
        assert pred_depth.shape[0] == batch_size

        pred = {
            'depth_pred_s0_b1hw': pred_depth,
        }
        return pred


@register_model(trainable=False)
def robust_mvd_5M(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    pretrained_weights = 'https://lmb.informatik.uni-freiburg.de/people/schroepp/weights/robustmvd.pt'
    weights = pretrained_weights if (pretrained and weights is None) else None
    model = build_model_with_cfg(model_cls=RobustMVD, weights=weights, train=train, num_gpus=num_gpus)
    return model


@register_model
def robust_mvd(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    pretrained_weights = 'https://lmb.informatik.uni-freiburg.de/people/schroepp/weights/robustmvd_600k.pt'
    weights = pretrained_weights if (pretrained and weights is None) else None
    model = build_model_with_cfg(model_cls=RobustMVD, weights=weights, train=train, num_gpus=num_gpus)
    return model
