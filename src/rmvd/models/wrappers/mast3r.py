import os.path as osp
import math

import torch
from torchvision import transforms as T
import torch.nn as nn
import numpy as np

from ..registry import register_model
from ..helpers import build_model_with_cfg
from rmvd.utils import get_path, get_torch_model_device, to_numpy, to_torch, select_by_index, exclude_index
from rmvd.data.transforms import ResizeInputs

CKPT_PATH =  '/mnt/nas3/shared/projects/fmvs/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'

class MAST3R_Wrapped(nn.Module):
    def __init__(self):
        super().__init__()

        import sys
        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
        repo_path = get_path(paths_file, "mast3r", "root")
        sys.path.insert(0, repo_path)
        sys.path.insert(0, repo_path + '/dust3r')
        from mast3r.model import AsymmetricMASt3R
        from dust3r.inference import inference

        model = AsymmetricMASt3R.from_pretrained(CKPT_PATH)
        model.eval()
        self.model = model
        self.inference = inference

        self.input_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        N = images[0].shape[0]

        previous_shape = images[0].shape[2:]

        # Resize to 336x448
        resized = ResizeInputs(size=(384, 512))({'images': images, 'intrinsics': intrinsics})
        images = resized['images']
        intrinsics = resized['intrinsics']

        # Input transform
        print(len(images))
        sds
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

                if idx == cur_keyview_idx:
                    pose = np.eye(4)
                else:
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
            min_depth = 0.25
            max_depth = 100
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
            'previous_shape': previous_shape
        }
        return sample

    def forward(
        self,
        images,
        intrinsics,
        poses,
        min_depth,
        max_depth,
        previous_shape,
        keyview_idx,
        **_
    ):

        # TODO: move this to input_adapter
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        images_source = images_source + [images_source[-1]] * max(0, (7 - len(images_source)))
        images_source = images_source[:7]
        images = [image_key] + images_source

        batch = [{'img': images[i], 'idx': i, 'instance': str(i)} for i in range(2)]
        pts1, pts2 = self.model(batch[0], batch[1])

        pred_depth = pts1['pts3d'][:, ..., 2].unsqueeze(1)

        pred = {
            'depth': pred_depth,
        }
        aux = {}

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)



class MAST3R_WrappedForMeshing(MAST3R_Wrapped):
    def forward(
        self,
        phase: str,
        cur_data: dict,
        src_data: dict,
        unbatched_matching_encoder_forward: bool,
        return_mask:bool,
    ):
        assert phase == 'test'

        batch = [
            {'img': cur_data['image_b3hw'], 'idx': 0, 'instance': '0'},
            {'img': src_data['image_b3hw'][:, 0], 'idx': 1, 'instance': '1'},
        ]
        pts1, pts2 = self.model(batch[0], batch[1])

        pred_depth = pts1['pts3d'][:, ..., 2].unsqueeze(1)

        pred = {
            'depth_pred_s0_b1hw': pred_depth,
        }
        return pred


@register_model(trainable=False)
def mast3r_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    cfg = {}
    model = build_model_with_cfg(model_cls=MAST3R_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model

