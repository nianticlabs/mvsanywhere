import os.path as osp
import math

import torch
from torchvision import transforms as T
import torch.nn as nn
import numpy as np
import kornia
from pathlib import Path

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
        from mast3r.fast_nn import fast_reciprocal_NNs

        self.fast_reciprocal_NNs = fast_reciprocal_NNs

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
        images = [image_key] + images_source

        poses_key = select_by_index(poses, keyview_idx)
        poses_source = exclude_index(poses, keyview_idx)

        intrinsics_key = select_by_index(intrinsics, keyview_idx)
        intrinsics_source = exclude_index(intrinsics, keyview_idx)

        all_preds = []
        all_confidences = []
        for i in range(len(images_source)):
            _poses_source = poses_source[i]
            _intrinsics_source = intrinsics_source[i]
            batch = [{'img': image_key, 'idx': 0, 'instance': str(0)}]
            batch.append({'img': images_source[i], 'idx': i, 'instance': str(i)})

            pts1, pts2 = self.model(batch[0], batch[1])

            pred_depth = pts1['pts3d'][:, ..., 2].unsqueeze(1)

            # triangulate the depth using GT cameras
            desc1, desc2 = pts1['desc'].squeeze(0).detach(), pts2['desc'].squeeze(0).detach()

            matches_im0, matches_im1 = self.fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                        device=pred_depth.device, dist='dot', block_size=2**13)

            poses_key_inv = torch.linalg.inv(poses_key.float())
            P1 = kornia.geometry.projection_from_KRt(
            intrinsics_key[..., :3, :3].float() * torch.tensor([4.0, 4.0, 1.0])[None, :, None].cuda(),
            poses_key_inv[..., :3, :3].float(),
            poses_key_inv[..., :3, 3:4].float()
            )
            poses_source_inv = torch.linalg.inv(_poses_source.float())
            P2 = kornia.geometry.projection_from_KRt(
            _intrinsics_source[..., :3, :3].float() * torch.tensor([4.0, 4.0, 1.0])[None, :, None].cuda(),
            poses_source_inv[..., :3, :3].float(),
            poses_source_inv[..., :3, 3:4].float()
            )

            matches_im0 = torch.tensor(matches_im0.copy()).unsqueeze(0).cuda()
            matches_im1 = torch.tensor(matches_im1.copy()).unsqueeze(0).cuda()
            matches_depth = kornia.geometry.epipolar.triangulate_points(
                P1, P2, matches_im0.float(), matches_im1.float()
            )[..., 2]

            # gather the predicted depth values at the matches
            xs, ys = matches_im0[..., 0], matches_im0[..., 1]
            pred_matches_depth = pred_depth[0, 0, ys, xs]

            scale = torch.median(matches_depth) / torch.median(pred_matches_depth)
            pred_depth = pred_depth * scale

            all_preds.append(pred_depth)
            all_confidences.append(pts1['conf'])

        all_preds = torch.stack(all_preds, dim=0)
        all_confidences = torch.stack(all_confidences, dim=0).unsqueeze(1)

        # torch.save(all_preds, 'all_preds.pth')
        # torch.save(all_confidences, 'all_confidences.pth')

        pred_depth = (all_preds * all_confidences).sum(dim=0) / all_confidences.sum(dim=0)

        pred = {
            'depth': pred_depth,
            'matches_depth': matches_depth,
            'matches_im0': matches_im0,
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

