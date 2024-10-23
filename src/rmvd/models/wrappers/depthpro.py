import os.path as osp
import math

import torch
from torchvision import transforms as T
import torch.nn as nn
import numpy as np
import json

from ..registry import register_model
from ..helpers import build_model_with_cfg
from rmvd.utils import get_path, get_torch_model_device, to_numpy, to_torch, select_by_index, exclude_index



class DepthPro_Wrapped(nn.Module):
    def __init__(self):
        super().__init__()

        import sys
        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
        repo_path = osp.join(get_path(paths_file, "depth_pro", "root"), "src")
        sys.path.insert(0, repo_path)
        from depth_pro.depth_pro import create_model_and_transforms, DepthProConfig

        CONFIG_DICT = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=osp.join(repo_path, "../checkpoints/depth_pro.pt"),
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )

        model, self.transform = create_model_and_transforms(config=CONFIG_DICT)
        model = model.eval()
        self.model = model

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        images = to_torch((images), device=device)

        sample = {
            'images': images,
            'keyview_idx': keyview_idx,
            'intrinsics': intrinsics,
        }

        # if intrinsics is not None:
        #     sample['fx'] = intrinsics[0][0]

        return sample

    def forward(self, images, keyview_idx, intrinsics):

        # TODO: move this to input_adapter
        image_key = select_by_index(images, keyview_idx)

        image_key = image_key[0].permute((1, 2, 0)).cpu().numpy()
        image = self.transform(image_key)
        image = image.cuda()

        with torch.inference_mode():
            # Using the provided intrinsics, which gives focal length in pixels
            pred = self.model.infer(image, f_px=torch.tensor(intrinsics[0][0, 0, 0]).cuda())

            # Predicted focallength_px is tensor(39216.5703, device='cuda:0') for  a 384x1280 image

        aux = {}

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def depthpro_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    cfg = {}
    model = build_model_with_cfg(model_cls=DepthPro_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model
