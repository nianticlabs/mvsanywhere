import os.path as osp

import torch
import torch.nn as nn

from ..registry import register_model
from ..helpers import build_model_with_cfg
from rmvd.utils import get_path, get_torch_model_device, to_numpy, to_torch, select_by_index


class UniDepth_Wrapped(nn.Module):

    def __init__(self):
        super().__init__()

        import sys
        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
        repo_path = get_path(paths_file, "unidepth", "root")
        sys.path.insert(0, repo_path)

        from unidepth.models import UniDepthV2

        name = "unidepth-v2-vitl14"
        self.model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
        self.model = self.model.to(torch.device("cuda"))

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        images = to_torch((images), device=device)

        sample = {
            'images': images,
            'keyview_idx': keyview_idx,
            'intrinsics': intrinsics,
        }

        return sample

    def forward(self, images, keyview_idx, intrinsics):

        # TODO: move this to input_adapter
        image_key = select_by_index(images, keyview_idx)

        rgb_input = image_key[0]
        intrinsics_input = torch.tensor(intrinsics[0][0])

        assert intrinsics_input.shape == (3, 3)
        assert rgb_input.shape[0] == 3

        with torch.inference_mode():
            # Using the provided intrinsics, which gives focal length in pixels
            predictions = self.model.infer(rgb_input, intrinsics_input)

        pred = {'depth': predictions['depth']}
        aux = {}

        assert len(pred['depth'].shape) == 4, pred['depth'].shape

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def unidepth_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    cfg = {}
    model = build_model_with_cfg(model_cls=UniDepth_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model
