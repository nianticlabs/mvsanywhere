"""
Regularised Splatfacto Model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Type, Optional, Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from nerfstudio.cameras.cameras import Cameras

# need following import for background color override
from nerfstudio.models.splatfacto import (SplatfactoModel, SplatfactoModelConfig, rasterization, renderers, 
                                          get_viewmat, random_quat_tensor, num_sh_bases, RGB2SH, Optimizers)
from nerfstudio.utils.rich_utils import CONSOLE

from regsplatfacto.utils import NormalGenerator


def get_points_on_sphere(N: int, radius: float = 250.0) -> torch.Tensor:
    """
    Generate N points on a sphere using the Fibonacci sphere algorithm.
    This is used to generate the points for the skybox gaussians.
    """
    # Generate indices
    indices = torch.arange(0, N, dtype=torch.float32) + 0.5

    # Compute angles
    phi = torch.acos(1 - 2 * indices / N)
    theta = torch.pi * (1 + 5 ** 0.5) * indices 

    # Convert to Cartesian coordinates
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)

    # Stack and scale by radius
    points = torch.stack((x, y, z), dim=1) * radius

    return points.float()   


@dataclass
class RegSplatfactoModelConfig(SplatfactoModelConfig):
    """RegSplatfacto Model Config - Splatfacto with depth and normal supervision"""

    _target: Type = field(default_factory=lambda: RegSplatfactoModel)

    use_scale_regularization: bool = True
    scale_regularisation_weight: float = 0.1
    """If scale regularization is enabled, a scale regularization introduced in PhysGauss
    (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians.
    We adapt their loss to use the ratio of max to median scale, which we found to work
    better at encouraging Gaussians to be disks.

    We have two config items to control this: use_scale_regularization and
    scale_regularisation_weight. Really we only use scale_regularisation_weight,
    but we keep use_scale_regularization as this is also used in the SplatfactoModelConfig.
    In __post_init__ we validate that these two config options are compatible.

    Note that the PhysGauss version is already implemented in SplatFacto; our implementation
    (with the max-to-median ratio) overwrites the existing output. So it is important that our
    code plays nicely with theirs, for example ensuring (a) we use the same key in the loss
    dictionary ("scale_reg"), and (b) like SplatFacto, we compute the loss every 10 steps.
    """
    max_gauss_ratio: float = 2.0
    """Threshold of ratio of Gaussian's max to median scale before applying regularization
    loss. This is adapted from the PhysGauss paper (there they used ratio of max to min;
    we have found that max to median does a better job of encouraging disk-shaped Gaussians.).
    """
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""

    regularisation_first_step: int = 0
    """First step to apply regularisation"""
    use_depth_loss_warmup: bool = True
    """If True, downweight depth loss for the first 1k steps"""
    depth_regularisation_weight: float = 0.0
    """Weight of the depth regularisation loss"""
    use_metric_depth_loss: bool = False
    """If True, use metric depth loss rather than scale invariant"""
    normal_regularisation_weight: float = 0.1
    """Weight of the normal regularisation loss"""
    flat_regularisation_weight: float = 1.0
    """Weight of the regularisation loss encouraging gaussians to be flat, i.e. set their minimum
    scale to be small"""
    mask_sky_for_normal_loss: bool = True
    """If True, mask sky regions for normal loss"""
    use_skybox: bool = False
    """If True, use a skybox to render the background"""
    skybox_gaussian_count: int = 10000
    """Number of gaussians to use for the skybox"""
    skybox_radius: float = 200.0
    """Radius of the skybox gaussians"""

    def __post_init__(self) -> None:
        if self.output_depth_during_training is False:
            raise ValueError("output_depth_during_training must be True for RegSplatfacto!")

        if self.max_gauss_ratio < 1.0:
            raise ValueError("max_gauss_ratio is not used if it is less than 1.0.")

        if self.use_scale_regularization and self.scale_regularisation_weight == 0.0:
            raise ValueError(
                "use_scale_regularization is True but scale_regularisation_weight is 0.0."
            )

        if not self.use_scale_regularization and self.scale_regularisation_weight > 0.0:
            raise ValueError(
                "use_scale_regularization is False but scale_regularisation_weight is greater "
                "than 0.0."
            )

        if self.flat_regularisation_weight == 0.0 and self.scale_regularisation_weight > 0.0:
            raise ValueError(
                "scale_regularisation_weight is greater than zero, but flat_regularisation_weight "
                "is zero. Scale regularisation will only be applied when "
                "flat_regularisation_weight is > 0.0"
            )


class RegSplatfactoModel(SplatfactoModel):
    """
    Nerfstudio Splatfacto model with added depth and normal regularisation.

    As most functionality is inherited from Splatfacto, most of the code in this class represents
    new functionality introduced in RegSplatfacto. The only exception to this is the
    'split_gaussians' method. We needed to add functionality in to the *middle* of the method,
    so we had to copy the code from the parent class into this class and modify it. The new code
    we added into that method is marked clearly.
    """

    config: RegSplatfactoModelConfig

    def populate_modules(self):
        super().populate_modules()

        # populate skybox gaussians
        skybox_means = get_points_on_sphere(
            self.config.skybox_gaussian_count, radius=self.config.skybox_radius
        )
        skybox_means = torch.nn.Parameter(skybox_means, requires_grad=False)

        distances, indices = self.k_nearest_sklearn(skybox_means.data, 3)
        distances = torch.from_numpy(distances)
        self.nearest_indices = torch.from_numpy(indices).long()
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True) #* 0.1
        skybox_scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = skybox_means.shape[0]
        skybox_quats = torch.nn.Parameter(random_quat_tensor(num_points))
        # dim_sh = num_sh_bases(self.config.sh_degree)

        skybox_features_dc = torch.nn.Parameter(torch.ones(num_points, 3) * 0.7)
        # skybox_features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        skybox_opacities = torch.nn.Parameter(torch.logit(0.5 * torch.ones(num_points, 1)))

        self.skybox_params = torch.nn.ParameterDict(
            {
                "skybox_means": skybox_means,
                "skybox_scales": skybox_scales,
                "skybox_quats": skybox_quats,
                "skybox_features_dc": skybox_features_dc,
                "skybox_opacities": skybox_opacities,
            }
        )

    def get_skybox_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            name: [self.skybox_params[name]]
            for name in ["skybox_scales", "skybox_quats", "skybox_features_dc", "skybox_opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        if self.config.use_skybox:
            gps.update(self.get_skybox_param_groups())
        return gps

    @property
    def skybox_means(self):
        return self.skybox_params["skybox_means"]

    @property
    def skybox_scales(self):
        return self.skybox_params["skybox_scales"]
    
    @property
    def skybox_quats(self):
        return self.skybox_params["skybox_quats"]
    
    @property
    def skybox_features_dc(self):
        return self.skybox_params["skybox_features_dc"]
    
    @property
    def skybox_opacities(self):
        return self.skybox_params["skybox_opacities"]

    def split_gaussians(self, split_mask, samps) -> dict[str, torch.Tensor]:
        """
        This function splits gaussians that are too large. It splits them such that they lie within
        the ellipse defined by the two largest axes of the gaussian.

        Most of this code is copied from Splatfacto. The code introduced in regSplatFacto is
        between "### RegSplatfacto Code ###" and "### End ###".
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(
            f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}"
        )
        centered_samples = torch.randn(
            (samps * n_splits, 3), device=self.device
        )  # Nx3 of axis-aligned scales
        scaled_samples = torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples

        ### RegSplatfacto Code ###
        # set smallest scale to 0 so that when we split we split in the plane
        if self.step >= self.config.regularisation_first_step:
            min_idx = torch.argmin(self.scales[split_mask].repeat(samps, 1), dim=-1)
            scaled_samples[torch.arange(samps * n_splits), min_idx] = 0.0
        ### End ###

        # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(
            dim=-1, keepdim=True
        )  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def get_rasterized_outputs(self, camera: Cameras) -> dict[str, torch.Tensor | list]:  # type: ignore
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)

        # get the background color
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)

        camera_scale_fac = 1.0 / self._get_downscale_factor()
        viewmat = get_viewmat(optimized_camera_to_world)
        W, H = int(camera.width[0] * camera_scale_fac), int(camera.height[0] * camera_scale_fac)
        self.last_size = (H, W)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        K = camera.get_intrinsics_matrices().cuda()
        K[:, :2, :] *= camera_scale_fac
        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            sh_degree_to_use = None

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        if self.config.use_skybox:
            # rasterize the skybox as the background
            backgroud_renders, background_alphas, background_info = rasterization(
                means=self.skybox_means,
                quats=self.skybox_quats / self.skybox_quats.norm(dim=-1, keepdim=True),
                scales=torch.exp(self.skybox_scales),
                opacities=torch.sigmoid(self.skybox_opacities).squeeze(-1),
                colors=self.skybox_features_dc,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=BLOCK_WIDTH,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode=render_mode,
                sh_degree=None,
                sparse_grad=False,
                absgrad=True,
                rasterize_mode=self.config.rasterize_mode,
            )
            background = backgroud_renders[:, ..., :3] + (1 - background_alphas) * self.background_color.to(self.device)
            background = torch.clamp(background, 0.0, 1.0)
        else:
            background = self.background_color.to(self.device)

        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]

        alpha = alpha[:, ...]
        rgb_raw = render[:, ..., :3]
        rgb = rgb_raw + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgb_raw = torch.clamp(rgb_raw, 0.0, 1.0)
        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        return {"rgb": rgb.squeeze(0), "depth": depth_im, "accumulation": alpha.squeeze(0), "background": background.squeeze(0),
                "rgb_raw": rgb_raw.squeeze(0)}  # type: ignore

    def get_outputs(self, camera: Cameras) -> dict[str, torch.Tensor | list]:  # type: ignore
        """
        A wrapper around SplatfactoModel.get_outputs that adds depth and normal outputs.

        Note that the normal map added here is:
            (a) the normal map computed from the depth map, rather than by rendering Gaussians.
                Computing the normals from the depth gave better regularisation results.
            (b) scaled to [0, 1] range, rather than [-1, 1] range. This is because the items in the
                dictionary returned from get_outputs are what are visualised in SplatFacto, and so
                we want the normal map to be in the [0, 1] range for visualisation.
                However, this means that the normals are no longer unit scaled, and any downstream
                functionality which use these normals need to re-scale them back to [-1, 1].
        """

        outputs = self.get_rasterized_outputs(camera=camera)

        assert isinstance(outputs["depth"], torch.Tensor)

        if outputs["depth"].dim() != 3:
            raise AssertionError(
                f"We expect outputs['depth'] to HxWxC, but got shape {outputs['depth'].shape}"
            )

        K_144 = self._get_intrinsics(camera)
        normal_hw3 = self._get_implied_normal_from_depth(
            depth_hw1=outputs["depth"], invK_144=torch.inverse(K_144)
        )

        outputs["intrinsics"] = K_144
        outputs["normal"] = normal_hw3

        return outputs

    def get_loss_dict(
        self,
        outputs: dict[str, torch.Tensor],
        batch,
        metrics_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Computes and returns the losses dictionary.

        Args:
            outputs (dict[str, torch.Tensor]): The output to compute the loss dict from.
            batch (dict[str, torch.Tensor]): A ground truth batch corresponding to outputs
            metrics_dict (dict[str, torch.Tensor]): A dictionary of metrics, some of which we can
                use for computing the loss.

        Returns:
            dict[str, torch.Tensor]: A dictionary of losses.
        """

        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Overwrite standard splatfacto scale loss to use ratio of max to middle
        # since with our flat loss, the smallest scale is always near 0
        if self.config.flat_regularisation_weight > 0.0:
            # We follow the original SplatFacto implementation here and
            #   (a) only apply this loss every 10 steps, and
            #   (b) call the loss "scale_reg". This is important, to ensure we overwrite the
            #           original value of this computed in super().get_loss_dict.
            # We shouldn't change either of these things, else we might inadvertently end up using
            # the SplatFacto version of the loss instead of ours.
            if self.config.scale_regularisation_weight > 0.0 and self.step % 10 == 0:
                loss_dict["scale_reg"] = (
                    self.config.scale_regularisation_weight
                    * self.compute_scale_regularisation_loss_median()
                )

        if self.step < self.config.regularisation_first_step:
            return loss_dict

        # Regularise the smallest scale to be small, to encourage the Gaussians to be disks rather
        # than balls.
        flat_loss = self.compute_flat_loss()

        height, width, _ = outputs["rgb"].shape
        gt_depth_hw1, gt_normal_hw3 = self._get_gt_depth_and_normal(
            batch=batch, height=height, width=width
        )

        # skybox losses
        if self.config.use_skybox:
            non_sky_pixels = (gt_depth_hw1 > 0.0).float()
    
            accum = outputs['accumulation']
            sky_loss = accum[~non_sky_pixels.bool()].mean()
            non_sky_loss = 1.0 - accum[non_sky_pixels.bool()].mean()
            loss_dict["sky_loss"] = sky_loss + non_sky_loss

            background = outputs["background"]
            non_sky_mask_expanded = non_sky_pixels.bool().squeeze(-1)
            mean_sky = background[~non_sky_mask_expanded].detach().mean(dim=0, keepdim=True)
            non_sky_color_loss = torch.abs(background[non_sky_mask_expanded] - mean_sky).mean() * 0.002
            loss_dict["sky_loss"] = loss_dict["sky_loss"] + non_sky_color_loss

            skybox_features_mean = self.skybox_features_dc.mean(dim=0, keepdim=True).detach()
            skybox_features_loss = torch.abs(self.skybox_features_dc - skybox_features_mean).mean() * 0.01
            loss_dict["skybox_loss"] = skybox_features_loss
        else:
            non_sky_pixels = torch.ones_like(gt_depth_hw1).float()

        # rescale both normals from [0, 1] back to [-1, 1] (i.e. unit vectors).
        # See note in _get_implied_normal_from_depth docstring for details.
        pred_normal_hw3 = (outputs["normal"] * 2.0) - 1.0
        gt_normal_hw3 = (gt_normal_hw3 * 2.0) - 1.0

        normal_loss = self.compute_normal_loss(pred_normal_hw3, gt_normal_hw3, non_sky_pixels)

        loss_dict["flat_loss"] = flat_loss * self.config.flat_regularisation_weight
        loss_dict["normal_loss"] = normal_loss * self.config.normal_regularisation_weight

        if self.config.depth_regularisation_weight > 0.0:
            depth_loss = self.compute_depth_loss(pred_depth=outputs["depth"], gt_depth=gt_depth_hw1)

            if self.config.use_depth_loss_warmup and self.step < 1000:
                loss_dict["depth_loss"] = depth_loss * 0.01
            else:
                loss_dict["depth_loss"] = depth_loss * self.config.depth_regularisation_weight
        else:
            loss_dict["depth_loss"] = torch.tensor(0.0, device=outputs["depth"].device)

        return loss_dict

    @lru_cache(maxsize=None)
    def _get_normal_generator(self, height: int, width: int) -> NormalGenerator:
        """
        Gets a normal generator object.

        This is wrapped in lru_cache so for a given height and width, we only create one instance
        of the normal generator during the whole lifetime of this class instance.

        Args:
            height (int): The height of the depth map.
            width (int): The width of the depth map.

        Returns:
            NormalGenerator: The normal generator object.
        """
        return NormalGenerator(height=height, width=width).cuda()

    def _get_intrinsics(self, camera: Cameras) -> torch.Tensor:
        """
        Returns the 1x4x4 unnormalised intrinsics matrix for a camera.

        Args:
            camera (Cameras): The nerfstudio camera object.

        Returns:
            torch.Tensor: The 1x4x4 unnormalised intrinsics matrix.
        """
        camera_scale_fac = 1.0 / self._get_downscale_factor()

        K_144 = torch.eye(4).unsqueeze(0).cuda()
        K_144[:, :3, :3] = camera.get_intrinsics_matrices().cuda()
        K_144[:, :2, :] *= camera_scale_fac

        return K_144

    def _get_implied_normal_from_depth(
        self, depth_hw1: torch.Tensor, invK_144: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the normal map from a depth map.

        Args:
            depth_hw1 (torch.Tensor): The depth map in HWC format.
            invK_144 (torch.Tensor): The inverse intrinsics matrix.

        Returns:
            torch.Tensor: The normal map in HWC format, scaled to [0, 1] range.
                This matches the expected shape and format for visualisation in SplatFacto.
                This needs to be rescaled to [-1, 1] when computing losses.
        """
        # Reshape depth to b1hw for normal generator
        height, width = depth_hw1.shape[:2]
        depth_11hw = depth_hw1.reshape(1, 1, height, width)

        # Estimate the normals from the depth map
        normal_13hw = self._get_normal_generator(height=height, width=width)(depth_11hw, invK_144)

        # Rescale normals from [-1, 1] to [0, 1]. See note in the docstring about the scaling.
        normal_13hw = 0.5 * (1.0 + normal_13hw)

        # reshape to match HWC convention of splatfacto
        normal_hw3 = normal_13hw.squeeze(0).permute(1, 2, 0)

        return normal_hw3

    def compute_scale_regularisation_loss_median(self) -> torch.Tensor:
        """
        Computes the scale regularisation loss as the ratio between the maximum and median
        scale of the Gaussians. This is only applied to Gaussians with ratios above
        self.config.max_gauss_ratio.

        This is adapted from the PhysGauss paper (https://xpandora.github.io/PhysGaussian/).
        In that paper, they used the ratio of max to min scale. We have found that max to median
        does a better job of encouraging disk-shaped Gaussians.

        Returns:
            torch.Tensor: The scale regularisation loss as a scalar.
        """
        # For each Gaussian, compute the ratio between the maximum and median (middle) dimension
        scale_exp = torch.exp(self.scales)
        ratio = scale_exp.amax(dim=-1) / scale_exp.median(dim=-1).values

        # Gaussians with ratios below max_gauss_ratio have no loss applied to them.
        # Gaussians with ratios above max_gauss_ratio have their ratio minimised.
        # The following diagram shows how the scale_reg loss varies as the ratio varies:
        #
        #           ▲
        #           │
        #           │       max_gauss_ratio       x
        #           │               │           x
        # scale_reg │               │         x
        #           │               │       x
        #           │               │     x
        #           │               │   x
        #           │               ▼ x
        #       0.0 └xxxxxxxxxxxxxxxx──────────────────►
        #           1.0
        #                          ratio
        #
        max_gauss_ratio = torch.tensor(self.config.max_gauss_ratio)
        scale_reg = torch.maximum(ratio, max_gauss_ratio) - max_gauss_ratio
        return scale_reg.mean()  # this has a weighting applied in get_loss_dict

    def compute_flat_loss(self) -> torch.Tensor:
        """
        Computes the flatness loss. This encourages the smallest scale of each Gaussian to be small.

        This should have the effect of encouraging Gaussians to be disks (or spikes) rather than
        balls. There is a separate `scale_regularisation_loss` which encourages the Gaussians to
        be disks rather than spikes.

        Returns:
            torch.Tensor: The flatness loss as a scalar.
        """
        flat_loss = torch.exp(self.scales).amin(dim=-1).mean()
        return flat_loss

    def compute_depth_loss(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
        """
        Computes the depth loss.

        Args:
            pred_depth (torch.Tensor): The predicted depth map.
            gt_depth (torch.Tensor): The ground truth depth map.

        Returns:
            torch.Tensor: The depth loss as a scalar.
        """
        if self.config.use_metric_depth_loss:
            depth_loss = self.compute_metric_depth_loss(pred_depth, gt_depth)
        else:
            depth_loss = self.compute_scale_invariant_depth_loss(pred_depth, gt_depth)
        return depth_loss

    def compute_metric_depth_loss(
        self, pred_depth: torch.Tensor, gt_depth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the metric depth loss.

        Args:
            pred_depth (torch.Tensor): The predicted depth map.
            gt_depth (torch.Tensor): The ground truth depth map.

        Returns:
            torch.Tensor: The metric depth loss as a scalar.
        """
        # log_diff = torch.log(pred_depth + 1e-6) - torch.log(gt_depth + 1e-6)
        # depth_loss = torch.abs(log_diff).mean()
        valid_pix = gt_depth > 0.0
        if valid_pix.sum() == 0:
            return torch.tensor(0.0, device=pred_depth.device)
        
        depth_loss = torch.abs(pred_depth[valid_pix] - gt_depth[valid_pix]).mean()
        # depth_loss += 0.1 * torch.abs(pred_depth[~valid_pix] - 100.).mean()
        return depth_loss

    def compute_scale_invariant_depth_loss(
        self, pred_depth: torch.Tensor, gt_depth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the scale invariant depth loss

        This is from Eqn. (4) from Eigen et al's "Depth Map Prediction from a Single Image using
        a Multi-Scale Deep Network", NeurIPS 2014. (https://arxiv.org/pdf/1406.2283)
        """
        assert pred_depth.shape == gt_depth.shape, f"Shapes of pred_depth and gt_depth do not match"

        log_diff = torch.log(pred_depth + 1e-6) - torch.log(gt_depth + 1e-6)
        depth_loss = torch.sqrt((log_diff**2).mean() - (log_diff.mean() ** 2))
        return depth_loss

    def compute_normal_loss(
        self, pred_normal_hw3: torch.Tensor, gt_normal_hw3: torch.Tensor, mask_hw1: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the loss between ground truth and predicted normals.
        This is computed as a combination of L1 and cosine loss.

        Args:
            pred_normal_hw3 (torch.Tensor): Predicted normal map in HWC format. These are expected
                to be unit vectors.
            gt_normal_hw3 (torch.Tensor): Ground truth normal map in HWC format. These are expected
                to be unit vectors.
            mask_hw1 (torch.Tensor): Mask for pixels to apply the loss to.

        Returns:
            torch.Tensor: The normal loss as a scalar.
        """
        total_unmasked = mask_hw1.sum() + 1e-5

        per_pixel_dot_product_hw1 = (gt_normal_hw3 * pred_normal_hw3).sum(dim=-1, keepdim=True)

        cosine_loss = 1 - (per_pixel_dot_product_hw1 * mask_hw1).sum() / total_unmasked
        return cosine_loss

    def _get_gt_depth_and_normal(
        self, batch: dict[str, torch.Tensor], height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the ground truth depth and normal from the batch and pre-processes them.

        Args:
            batch (dict[str, torch.Tensor]): The batch of data.
            height (int): The desired height of the depth and normal maps.
            width (int): The desired width of the depth and normal maps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The ground truth depth and normal maps as
                height x width x 1 and height x width x 3 tensors respectively.
        """
        gt_depth_1hw = batch["depth"].to(self.device)
        gt_normal_3hw = batch["normal"].to(self.device)

        assert (
            gt_depth_1hw.shape[0] == 1
        ), f"Expected depth to be of shape 1xhxw, got {gt_depth_1hw.shape}"
        assert (
            gt_normal_3hw.shape[0] == 3
        ), f"Expected normal to be of shape 3xhxw, got {gt_normal_3hw.shape}"

        gt_depth_1hw = F.interpolate(
            gt_depth_1hw.unsqueeze(0), (height, width), mode="bilinear", align_corners=False
        ).squeeze(0)

        gt_normal_3hw = F.interpolate(
            gt_normal_3hw.unsqueeze(0), (height, width), mode="bilinear", align_corners=False
        ).squeeze(0)

        # reshape to match HWC format of splatfacto
        gt_depth_hw1 = gt_depth_1hw.reshape((height, width, 1))
        gt_normal_hw3 = gt_normal_3hw.permute(1, 2, 0)

        return gt_depth_hw1, gt_normal_hw3
