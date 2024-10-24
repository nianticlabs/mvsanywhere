import os.path as osp

import torch
import torch.nn as nn

from ..registry import register_model
from ..helpers import build_model_with_cfg
from rmvd.utils import get_path, get_torch_model_device, to_numpy, to_torch, select_by_index


class Metric3D_Wrapped(nn.Module):

    # TODO - might not need these if we use metric3d functions
    padding_values = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1)

    def __init__(self):
        super().__init__()

        import sys
        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
        repo_path = get_path(paths_file, "metric3d", "root")
        sys.path.insert(0, repo_path)
        from mono.utils.do_test import resize_for_input, transform_test_data_scalecano

        self.resize_for_input = resize_for_input
        self.transform_test_data_scalecano = transform_test_data_scalecano

        print(10*"\nWARNING – should switch to vit_large")
        model_name: str = "metric3d_vit_small"
        self.model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=True)#.cuda()

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        images = to_torch((images), device=device)

        sample = {
            'images': images,
            'keyview_idx': keyview_idx,
            'intrinsics': intrinsics,
        }

        return sample

    def _standardize_image(self, image_b3hw: torch.Tensor) -> torch.Tensor:
        """
        Standardize the image to have a mean of 0 and std of 1
        """
        assert len(image_b3hw.shape) == 4, image_b3hw.shape
        assert image_b3hw.shape[1] == 3, image_b3hw.shape
        mean = self.padding_values.to(image_b3hw.device)
        std = self.std.to(image_b3hw.device)
        return (image_b3hw - mean) / std

    def forward(self, images, keyview_idx, intrinsics):

        # TODO: move this to input_adapter
        image_key = select_by_index(images, keyview_idx)

        # pre-process image (TODO – use the metric3d functions for this)
        image = torch.Tensor(image_key)
        image = self._standardize_image(image)

        if intrinsics is not None:
            f_x = torch.tensor(intrinsics[0][0, 0, 0]).cuda()
        else:
            f_x = None

        with torch.inference_mode():
            # Using the provided intrinsics, which gives focal length in pixels
            pred_depth, _, output_dict = self.model.inference({"input": image})

        pred = {}
        pred['depth'] = pred_depth
        aux = {}

        assert len(pred['depth'].shape) == 4, pred['depth'].shape

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def metric3d_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    cfg = {}
    model = build_model_with_cfg(model_cls=Metric3D_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model
