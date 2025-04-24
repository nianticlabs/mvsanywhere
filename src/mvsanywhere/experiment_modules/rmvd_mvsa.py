import sys
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import transforms as T

import kornia
import numpy as np

from mvsanywhere.utils.model_utils import get_model_class, load_model_inference

from rmvd.utils import get_path, get_torch_model_device, to_numpy, to_torch, select_by_index, exclude_index
from rmvd.data.transforms import ResizeInputs

class MVSA_Wrapped(nn.Module):
    def __init__(self, opts, use_refinement=False, **kwargs):
        super().__init__()

        model_class_to_use = get_model_class(opts)
        model = load_model_inference(opts, model_class_to_use)
        model = model.eval()
        self.model = model
        self.name = "MVSAnywhere"

        self.use_refinement = use_refinement

        self.input_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.min_depth_cache = {}
        self.max_depth_cache = {}

        
    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        # Input transform
        for idx, image_batch in enumerate(images):
            tmp_images = []
            image_batch = image_batch.transpose(0, 2, 3, 1)
            for image in image_batch:
                image = self.input_transform(image.astype(np.uint8)).float()
                tmp_images.append(image)

            image_batch = torch.stack(tmp_images)
            images[idx] = image_batch

        poses_list = []
        intrinsic_list = []
        for idx, (intrinsic_batch, pose_batch) in enumerate(zip(intrinsics, poses)):
            poses_batch_list = []
            intrinsic_batch_list = []
            for intrinsic, pose, cur_keyview_idx in zip(intrinsic_batch, pose_batch, keyview_idx):

                pose = np.linalg.inv(pose)

                intrinsic[0, :] *= 0.25
                intrinsic[1, :] *= 0.25

                K = np.eye(4)
                K[:3, :3] = intrinsic

                poses_batch_list.append(pose)
                intrinsic_batch_list.append(K)

            poses_list.append(np.stack(poses_batch_list))
            intrinsic_list.append(np.stack(intrinsic_batch_list))
        
        if depth_range is None:
            min_depth = None
            max_depth = None
        else:
            min_depth, max_depth = depth_range

        images, keyview_idx, intrinsic_list, poses_list = \
            to_torch((
                images,
                keyview_idx,
                intrinsic_list,
                poses_list,
            ), device=device)

        sample = {
            'images': images,
            'keyview_idx': keyview_idx,
            'intrinsics': intrinsic_list,
            'poses': poses_list,
            'min_depth': min_depth,
            'max_depth': max_depth,
        }
        return sample

    def forward(
        self,
        images,
        intrinsics,
        poses,
        min_depth,
        max_depth,
        keyview_idx,
        **_
    ):
        with torch.inference_mode():
            # TODO: move this to input_adapter
            image_key = select_by_index(images, keyview_idx)
            images_source = exclude_index(images, keyview_idx)

            poses_key = select_by_index(poses, keyview_idx)
            poses_source = exclude_index(poses, keyview_idx)

            intrinsics_key = select_by_index(intrinsics, keyview_idx)
            intrinsics_source = exclude_index(intrinsics, keyview_idx)
            
            cur_data = {
                "image_b3hw": image_key.float(),
                "cam_T_world_b44": torch.linalg.inv(poses_key).float(),
                "world_T_cam_b44": poses_key.float(),
                "K_matching_b44": intrinsics_key.float(),
                "invK_matching_b44": torch.linalg.inv(intrinsics_key).float(),
            }

            if min_depth is not None and max_depth is not None:
                cur_data["min_depth"] = min_depth
                cur_data["max_depth"] = max_depth

            src_data = {
                "image_b3hw": torch.stack(images_source, dim=1).float(),
                "cam_T_world_b44": torch.linalg.inv(torch.stack(poses_source, dim=1)).float(),
                "world_T_cam_b44": torch.stack(poses_source, dim=1).float(),
                "K_matching_b44": torch.stack(intrinsics_source, dim=1).float(),
                "invK_matching_b44": torch.linalg.inv(torch.stack(intrinsics_source, dim=1)).float()
            }
            
            outputs = self.model(
                phase="test",
                cur_data=cur_data,
                src_data=src_data,
                unbatched_matching_encoder_forward=True,
                return_mask=True,
                num_refinement_steps=int(self.use_refinement),
            )

            pred_depth = outputs["depth_pred_s0_b1hw"]

            pred = {
                'depth': pred_depth,
            }
            aux = {}

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)

