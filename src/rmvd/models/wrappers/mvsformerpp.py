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

CKPT_PATH = '/mnt/nas3/shared/projects/fmvs/MVSFormerPlusPlus/weights/model_best.pth'
CONFIG_PATH = '/mnt/nas3/shared/projects/fmvs/MVSFormerPlusPlus/config/mvsformer++.json'

class MVSFormerPP_Wrapped(nn.Module):
    def __init__(self, sample_in_inv_depth_space=False, num_sampling_steps=192):
        super().__init__()

        import sys
        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
        repo_path = get_path(paths_file, "mvsformerpp", "root")
        sys.path.insert(0, repo_path)
        from models.networks.DINOv2_mvsformer_model import DINOv2MVSNet

        with open(CONFIG_PATH) as f:
            config = json.load(f)

        model = DINOv2MVSNet(config['arch']['args'])

        checkpoint = torch.load(CKPT_PATH)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, val in state_dict.items():
            if "pe_dict" in key:
                continue
            key_ = key[7:] if key.startswith("module.") else key
            new_state_dict[key_] = val
        model.load_state_dict(new_state_dict, strict=True)

        # prepare models for testing
        model.eval()
        self.model = model

        
        self.sample_in_inv_depth_space = sample_in_inv_depth_space
        self.num_sampling_steps = num_sampling_steps

        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        N = images[0].shape[0]
        orig_ht, orig_wd = images[0].shape[-2:]
        ht, wd = int(math.ceil(orig_ht / 64.0) * 64.0), int(math.ceil(orig_wd / 64.0) * 64.0)
        if (orig_ht != ht) or (orig_wd != wd):
            resized = ResizeInputs(size=(ht, wd))({'images': images, 'intrinsics': intrinsics})
            images = resized['images']
            intrinsics = resized['intrinsics']

        for idx, image_batch in enumerate(images):
            tmp_images = []
            image_batch = image_batch.transpose(0, 2, 3, 1)
            for image in image_batch:
                image = self.input_transform(image.astype(np.uint8)).float()
                tmp_images.append(image)

            image_batch = torch.stack(tmp_images)
            images[idx] = image_batch

        proj_mats = []
        for idx, (intrinsic_batch, pose_batch) in enumerate(zip(intrinsics, poses)):
            proj_mat_batch = []
            for intrinsic, pose, cur_keyview_idx in zip(intrinsic_batch, pose_batch, keyview_idx):

                scale_arr = np.array([[0.25] * 3, [0.25] * 3, [1.] * 3])  # 3, 3
                intrinsic = intrinsic * scale_arr  # scale intrinsics to 4x downsampling that happens within the model

                if idx == cur_keyview_idx:
                    pose = np.eye(4)
                else:
                    pose = pose

                # pose[:3, 3] *= 1000.0

                proj_mat = np.zeros((2, 4, 4))
                proj_mat[0, :4, :4] = pose
                proj_mat[1, :3, :3] = intrinsic

                proj_mat_batch.append(proj_mat)

            proj_mat_batch = np.stack(proj_mat_batch)
            proj_mats.append(proj_mat_batch)
        
        if depth_range is None:
            min_depth, max_depth = 0.0, 100.0
        else:
            min_depth, max_depth = depth_range

        # min_depth *= 1000.0
        # max_depth *= 1000.0
        depth_interval = (max_depth - min_depth) / 192
        depth_interval = depth_interval * 1.06
        depth_values = np.arange(min_depth, depth_interval * (192 - 0.5) + min_depth, depth_interval, dtype=np.float32)

        images, keyview_idx, proj_mats, depth_values = \
            to_torch((images, keyview_idx, proj_mats, depth_values), device=device)

        sample = {
            'images': images,
            'keyview_idx': keyview_idx,
            'proj_mats': proj_mats,
            'depth_values': depth_values,
        }
        return sample

    def forward(self, images, proj_mats, depth_values, keyview_idx, **_):

        # TODO: move this to input_adapter
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        images = [image_key] + images_source

        proj_mat_key = select_by_index(proj_mats, keyview_idx)
        proj_mats_source = exclude_index(proj_mats, keyview_idx)
        proj_mats = [proj_mat_key] + proj_mats_source

        images = torch.stack(images, 1)  # N, num_views, 3, H, W
        proj_mats = torch.stack(proj_mats, 1).float()  # N, num_views, 2, 4, 4

        stage0_pjmats = proj_mats.clone()
        stage0_pjmats[:, :, 1, :2, :] = proj_mats[:, :, 1, :2, :] * 0.5
        stage1_pjmats = proj_mats.clone()
        stage2_pjmats = proj_mats.clone()
        stage2_pjmats[:, :, 1, :2, :] = proj_mats[:, :, 1, :2, :] * 2
        stage3_pjmats = proj_mats.clone()
        stage3_pjmats[:, :, 1, :2, :] = proj_mats[:, :, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1": stage0_pjmats,
            "stage2": stage1_pjmats,
            "stage3": stage2_pjmats,
            "stage4": stage3_pjmats
        }

        depth_values = depth_values.unsqueeze(0)
        with torch.inference_mode():
            outputs = self.model.forward(images, proj_matrices_ms, depth_values, tmp=[5., 5., 5., 1.])
        pred_depth = outputs["refined_depth"]

        pred_depth = pred_depth.unsqueeze(1)

        pred = {
            'depth': pred_depth,
            # 'depth_uncertainty': pred_depth_uncertainty
        }
        aux = {}

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def mvsformerpp_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    cfg = {"sample_in_inv_depth_space": False, "num_sampling_steps": 192}
    model = build_model_with_cfg(model_cls=MVSFormerPP_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model
