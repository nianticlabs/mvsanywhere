import sys
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import transforms as T

import kornia
import numpy as np

from ..registry import register_model
from ..helpers import build_model_with_cfg
from rmvd.utils import get_path, get_torch_model_device, to_numpy, to_torch, select_by_index, exclude_index
from rmvd.data.transforms import ResizeInputs


DEFAULT_MAX_DEPTH = 100
DEFAULT_MIN_DEPTH = 0.25


def get_epipolar_depth_range(
        num_sources,
        image_key, images_source,
        poses_key, poses_source,
        intrinsics_key, intrinsics_source,
        threshold=0.05
    ):
    # Get fundamental matrix
    poses_key_inv = torch.linalg.inv(poses_key.float())
    P1 = kornia.geometry.projection_from_KRt(
        intrinsics_key[..., :3, :3].float() * torch.tensor([4.0, 4.0, 1.0])[None, :, None].cuda(),
        poses_key_inv[..., :3, :3].float(),
        poses_key_inv[..., :3, 3:4].float()
    )
    poses_source_inv = torch.linalg.inv(torch.cat(poses_source[:num_sources], dim=0).float())
    P2 = kornia.geometry.projection_from_KRt(
        torch.cat(intrinsics_source[:num_sources], dim=0)[..., :3, :3].float() * torch.tensor([4.0, 4.0, 1.0])[None, :, None].cuda(),
        poses_source_inv[..., :3, :3].float(),
        poses_source_inv[..., :3, 3:4].float()
    )
    Fm = kornia.geometry.epipolar.fundamental_from_projections(
        P1.expand(num_sources, -1, -1),
        P2
    )
    
    _, _, H, W = image_key.shape

    # All pixels
    xx, yy = torch.meshgrid(
        torch.arange(W),
        torch.arange(H),
        indexing="xy",
    )
    pix_coords_n2 = (torch.stack((xx, yy), axis=0) + 0.5).flatten(1).T.cuda()

    # Borders of the image
    border_lines = torch.stack([
        torch.tensor([1, 0, 0], dtype=torch.float32).cuda(),   # Left border (x = 0)
        torch.tensor([1, 0, -W], dtype=torch.float32).cuda(),  # Right border (x = width)
        torch.tensor([0, 1, 0], dtype=torch.float32).cuda(),   # Top border (y = 0)
        torch.tensor([0, 1, -H], dtype=torch.float32).cuda()  # Bottom border (y = height)
    ]).unsqueeze(1)

    # Matching
    min_depth = []
    max_depth = []
    for i in range(num_sources):
        
        
        # Compute epipolar lines
        epipolar_lines = kornia.geometry.epipolar.compute_correspond_epilines(
            pix_coords_n2,
            Fm[i:i+1]
        )

        # Intersections between epipolar lines and border of the image
        intersections = kornia.geometry.convert_points_from_homogeneous(
            torch.cross(border_lines, epipolar_lines)
        )

        # Intersections within the border of the image
        mask = (
            (intersections[..., 0] >= -1e-4) &
            (intersections[..., 0] <= W+1e-4) &
            (intersections[..., 1] >= -1e-4) &
            (intersections[..., 1] <= H+1e-4)
        )
        valid_pixels = mask.sum(dim=0) == 2

        # Compute the two intersections of each line
        mask = mask[:, valid_pixels]
        intersections = intersections[:, valid_pixels]

        indices = torch.nonzero(mask, as_tuple=False)
        intersections = intersections[indices[:, 0], indices[:, 1]][torch.argsort(indices[:, 1])].view(intersections.shape[1], 2, 2)

        epipolar_lines_distances = torch.linalg.norm(intersections[:, 0] - intersections[:, 1], dim=-1)

        # Point on the src image
        dst_pts = pix_coords_n2[valid_pixels][epipolar_lines_distances.argmax()][None, None, :].expand(-1, 64, -1)
        
        # Points along epipolar line on the cur image
        a, b = intersections[epipolar_lines_distances.argmax()]
        src_pts = (a + torch.linspace(0, 1, steps=64).unsqueeze(1).cuda() * (b - a)).unsqueeze(0)


        depths = kornia.geometry.epipolar.triangulate_points(
            P1, P2[i:i+1], src_pts.cuda(), dst_pts.cuda()
        )[..., 2]

        mask = (depths > 0.0)
        if mask.sum() == 0:
            continue

        pts_min_depth = torch.quantile(depths[mask], 0.05).item()
        pts_max_depth = torch.quantile(depths[mask], 0.95).item()
        
        if 0.0 < pts_min_depth:
            min_depth.append(pts_min_depth)
            max_depth.append(pts_max_depth)

    min_depth = np.median(min_depth) if len(min_depth) > 0 else None
    max_depth = np.median(max_depth) if len(max_depth) > 0 else None

    return min_depth, max_depth


def get_depth_range(
        num_sources,
        image_key, images_source,
        poses_key, poses_source,
        intrinsics_key, intrinsics_source,
        threshold=0.05
    ):
    # Get fundamental matrix
    poses_key_inv = torch.linalg.inv(poses_key.float())
    P1 = kornia.geometry.projection_from_KRt(
        intrinsics_key[..., :3, :3].float() * torch.tensor([4.0, 4.0, 1.0])[None, :, None].cuda(),
        poses_key_inv[..., :3, :3].float(),
        poses_key_inv[..., :3, 3:4].float()
    )
    poses_source_inv = torch.linalg.inv(torch.cat(poses_source[:num_sources], dim=0).float())
    P2 = kornia.geometry.projection_from_KRt(
        torch.cat(intrinsics_source[:num_sources], dim=0)[..., :3, :3].float() * torch.tensor([4.0, 4.0, 1.0])[None, :, None].cuda(),
        poses_source_inv[..., :3, :3].float(),
        poses_source_inv[..., :3, 3:4].float()
    )
    Fm = kornia.geometry.epipolar.fundamental_from_projections(
        P1.expand(num_sources, -1, -1),
        P2
    )
    
    # Extract features
    sift = kornia.feature.SIFTFeature(device='cuda')
    laf, resp, desc = sift(kornia.color.rgb_to_grayscale(torch.cat([image_key] + images_source[:num_sources], dim=0)))

    # Matching
    min_depth = []
    max_depth = []
    for i in range(num_sources):
        scores, matches = kornia.feature.match_snn(desc[0], desc[i+1], 0.9)

        if len(matches) == 0:
            continue

        src_pts = laf[0,matches[:,0], :, 2]
        dst_pts = laf[i+1,matches[:,1], :, 2]

        dists = kornia.geometry.epipolar.symmetrical_epipolar_distance(
                src_pts.unsqueeze(0), dst_pts.unsqueeze(0), Fm[i:i+1]
        )
        mask = dists < threshold

        if mask.sum() == 0:
            continue

        depths = kornia.geometry.epipolar.triangulate_points(
            P1, P2[i:i+1], src_pts[mask[0]][None], dst_pts[mask[0]][None]
        )[..., 2]

        mask = (depths > 0.0)
        if mask.sum() == 0:
            continue
        pts_min_depth = torch.quantile(depths[mask], 0.05).item()
        pts_max_depth = torch.quantile(depths[mask], 0.95).item()
        
        if 0.0 < pts_min_depth:
            min_depth.append(pts_min_depth * 0.25)
            max_depth.append(pts_max_depth * 2.0)

    min_depth = np.median(min_depth) if len(min_depth) > 0 else None
    max_depth = np.median(max_depth) if len(max_depth) > 0 else None

    return min_depth, max_depth


@dataclass
class SimplifiedOpts:
    model_type: str = "depth_model"
    load_weights_from_checkpoint: str = ""
    fast_cost_volume: bool = False


class FMVS_Wrapped(nn.Module):
    def __init__(self, code_dir, load_weights_from_checkpoint, fast_cost_volume=False):
        super().__init__()

        # Add code to path
        sys.path.insert(0, code_dir)
        from doubletake.utils.model_utils import get_model_class, load_model_inference

        opts = SimplifiedOpts(
            load_weights_from_checkpoint=load_weights_from_checkpoint,
            fast_cost_volume=fast_cost_volume
        )
    
        model_class_to_use = get_model_class(opts)
        model = load_model_inference(opts, model_class_to_use)
        model = model.eval()
        self.model = model

        

        self.input_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.min_depth_cache = {}
        self.max_depth_cache = {}

        
    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None, indices=None):
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
            min_depth = DEFAULT_MIN_DEPTH
            max_depth = DEFAULT_MAX_DEPTH
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
            'indices': indices,
            
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
        indices,
        **_
    ):

        # TODO: move this to input_adapter
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        images_source = images_source + [images_source[-1]] * max(0, (7 - len(images_source)))
        images_source = images_source[:7]

        poses_key = select_by_index(poses, keyview_idx)
        poses_source = exclude_index(poses, keyview_idx)
        poses_source = poses_source + [poses_source[-1]] * max(0, (7 - len(poses_source)))
        poses_source = poses_source[:7]

        intrinsics_key = select_by_index(intrinsics, keyview_idx)
        intrinsics_source = exclude_index(intrinsics, keyview_idx)
        intrinsics_source = intrinsics_source + [intrinsics_source[-1]] * max(0, (7 - len(intrinsics_source)))
        intrinsics_source = intrinsics_source[:7]

        self.model.depth_decoder.output_convs[0][1].size = tuple(image_key.shape[-2:])

        # Use sift max_depth
        if len(indices) == 1:
            # Build the cache
            min_depth, max_depth = get_epipolar_depth_range(
                len(images) - 1, image_key, images_source,
                poses_key, poses_source, intrinsics_key, intrinsics_source,
        )

            self.min_depth_cache[indices[0]] = min_depth
            self.max_depth_cache[indices[0]] = max_depth

        else:
            # use the cache
            min_depths, max_depths = [], []
            for idx in indices:
                min_depth = self.min_depth_cache.get(idx, None)
                max_depth = self.max_depth_cache.get(idx, None)
                if min_depth is not None:
                    min_depths.append(min_depth)
                if max_depth is not None:
                    max_depths.append(max_depth)

            if len(min_depths) == 0:
                min_depth = None
            else:
                min_depth = np.median(min_depths)
            if len(max_depths) == 0:
                max_depth = None
            else:
                max_depth = np.median(max_depths)

        # If sift didn't work, use simple heuristic
        cam_dists = torch.stack([torch.linalg.norm((poses_key.float() @ ps)[..., :3, 3], dim=1) for ps in poses_source])
        baseline_min_depth = (torch.min(cam_dists) * intrinsics_key[:, 0, 0]) * 4 / (1.0 * image_key.shape[-1])
        baseline_max_depth = (torch.max(cam_dists) * intrinsics_key[:, 0, 0] * 4 ) / (0.01 * image_key.shape[-1])
        if min_depth is None:
            min_depth = baseline_min_depth
        if max_depth is None:
            max_depth = baseline_max_depth

        print(min_depth, max_depth)
        
        # print(f"min_depth: {min_depth}, max_depth: {max_depth}", intrinsics_key[:, 0, 0] * 4, image_key.shape[-1])

        cur_data = {
            "image_b3hw": image_key.float(),
            "cam_T_world_b44": torch.linalg.inv(poses_key).float(),
            "world_T_cam_b44": poses_key.float(),
            "K_matching_b44": intrinsics_key.float(),
            "invK_matching_b44": torch.linalg.inv(intrinsics_key).float(),
            "min_depth": torch.tensor(min_depth).unsqueeze(0).float(),
            "max_depth": torch.tensor(max_depth).unsqueeze(0).float(),
        }

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


@register_model(trainable=False)
def fmvs_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    code_dir, load_weights_from_checkpoint = weights.split(':')
    cfg = {
        'code_dir': code_dir,
        'load_weights_from_checkpoint': load_weights_from_checkpoint
    }
    model = build_model_with_cfg(model_cls=FMVS_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model
