import os.path as osp

import torch
import torch.nn as nn

from ..registry import register_model
from ..helpers import build_model_with_cfg
from rmvd.utils import get_path, get_torch_model_device, to_numpy, to_torch, select_by_index


class DataBasic:
    # Attributes copied from Metric3D Readme
    # Setting these here to avoid having to load a config
    crop_size = (512, 1088)

    def __getitem__(self, key):
        if key == "canonical_space":
            return {'focal_length': 1000.0, 'img_size': (512, 960)}
        else:
            raise KeyError(f"Unknown key {key}")



class Metric3D_Wrapped(nn.Module):

    def __init__(self):
        super().__init__()

        import sys
        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
        repo_path = get_path(paths_file, "metric3d", "root")
        sys.path.insert(0, repo_path)
        from mono.utils.do_test import resize_for_input, transform_test_data_scalecano

        self.resize_for_input = resize_for_input
        self.transform_test_data_scalecano = transform_test_data_scalecano

        # From the Metric3D readme:
        self.data_basic = DataBasic()

        print(10*"\nWARNING – should switch to vit_large")
        model_name: str = "metric3d_vit_small"
        self.model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=True)#
        self.model = self.model.cuda()

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

        # pre-process image (TODO – use the metric3d functions for this)
        image = torch.Tensor(image_key)[0].permute(1, 2, 0).cpu().numpy()  # -> (H, W, 3)
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        # image = self._standardize_image(image)

        K = intrinsics[0][0]
        assert K.shape == (3, 3)

        metric3d_intrinsic = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        # let's do the metric3d preprocessing
        rgb_input, _, pad, label_scale_factor = self.transform_test_data_scalecano(rgb=image, intrinsic=metric3d_intrinsic, data_basic=self.data_basic)

        with torch.inference_mode():
            # Using the provided intrinsics, which gives focal length in pixels
            pred_depth, _, output_dict = self.model.inference({"input": rgb_input[None, ...]})

        print("Label scale factor (should be dividing by this)", label_scale_factor)

        # This step follows the `pred_depth = pred_depth * normalize_scale / scale_info`
        # step in Metric3D's postprocess_per_image function
        pred_depth = pred_depth / label_scale_factor

        # Undo the padding
        _, _, h, w = pred_depth.shape
        pred_depth = pred_depth[:, :, pad[0] : h - pad[1], pad[2] : w - pad[3]]

        pred = {'depth': pred_depth}
        aux = {}

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
