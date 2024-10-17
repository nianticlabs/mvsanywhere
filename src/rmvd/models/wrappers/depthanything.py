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
from rmvd.data.transforms import ResizeInputs

CKPT_PATH = '/mnt/nas3/shared/projects/fmvs/Depth-Anything-V2/weights/depth_anything_v2_vitb.pth'
# CKPT_PATH = '/mnt/nas3/shared/projects/fmvs/Depth-Anything-V2/metric_depth/depth_anything_v2_metric_vkitti_vitl.pth'
# CKPT_PATH = '/mnt/nas3/shared/projects/fmvs/Depth-Anything-V2/metric_depth/depth_anything_v2_metric_hypersim_vitl.pth'


class DepthAnything_Wrapped(nn.Module):
    def __init__(self):
        super().__init__()

        import sys
        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
        repo_path = get_path(paths_file, "depthanything", "root")
        sys.path.insert(0, repo_path)
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

        # model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': 80.0})
        model = DepthAnythingV2(**{**model_configs[encoder]})
        model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
        model = model.eval()
        self.model = model

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        N = images[0].shape[0]

        images = to_torch((images), device=device)

        sample = {
            'images': images,
            'keyview_idx': keyview_idx,
        }
        return sample

    def forward(self, images, keyview_idx, **_):

        # TODO: move this to input_adapter
        image_key = select_by_index(images, keyview_idx)

        image_key = image_key[0].permute((1, 2, 0)).cpu().numpy()
        image_key = image_key[..., ::-1]

        with torch.inference_mode():
            o = self.model.infer_image(image_key)

        pred = {
            'depth': torch.tensor(np.divide(1.0, o, out=np.zeros_like(o), where=o != 0)).unsqueeze(0).unsqueeze(0),
            # 'depth': torch.tensor(o).unsqueeze(0).unsqueeze(0),
            # 'depth_uncertainty': pred_depth_uncertainty
        }
        aux = {}

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def depthanything_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    cfg = {}
    model = build_model_with_cfg(model_cls=DepthAnything_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model
