import logging

import lightning as pl
import timm
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import numpy as np

from mvsanywhere.losses import (
    MSGradientLoss,
    MVDepthLoss,
    NormalsLoss,
    ScaleInvariantLoss,
)
from mvsanywhere.modules.cost_volume import CostVolumeManager, FastCostVolumeManager
from mvsanywhere.modules.depth_anything_blocks import DPTHead
from mvsanywhere.modules.feature_volume import FastFeatureVolumeManager, FeatureVolumeManager
from mvsanywhere.modules.view_agnostic_feature_volume import FastViewAgnosticFeatureVolumeManager, ViewAgnosticFeatureVolumeManager
from mvsanywhere.modules.layers import TensorFormatter
from mvsanywhere.modules.networks import (
    CVEncoder,
    DepthDecoderPP,
    ResnetMatchingEncoder,
    UNetMatchingEncoder,
)
from mvsanywhere.modules.networks_fast import SkipDecoderRegression
from mvsanywhere.modules.vit_modules import DINOv2, ViTCVEncoder
from mvsanywhere.utils.augmentation_utils import CustomColorJitter
from mvsanywhere.utils.generic_utils import (
    reverse_imagenet_normalize,
    tensor_B_to_bM,
    tensor_bM_to_B,
)
from mvsanywhere.utils.geometry_utils import NormalGenerator
from mvsanywhere.utils.metrics_utils import compute_depth_metrics
from mvsanywhere.utils.visualization_utils import colormap_image

logger = logging.getLogger(__name__)


class DepthModel(pl.LightningModule):
    """Class for SimpleRecon depth estimators.

    This class handles training and inference for SimpleRecon models.

    Depth maps will be predicted

    It houses experiments for a vanilla cost volume that uses dot product
    reduction and the full feature volume.

    It also allows for experimentation on the type of image encoder.

    The opts the model is first initialized with will be saved as part of
    hparams. On load from checkpoint the model will use those stores
    options. It's generally a good idea to use those directly, unless you
    want to do something fancy with changing resolution, since some
    projective modules use initialized spatial grids for
    projection/backprojection.

    Attributes:
        run_opts: options object with flags.
        matching_model: the model used for generating image features for
            matching in the cost volume.
        cost_volume: the cost volume module.
        encoder: the image encoder used to enforce an image prior.
        cost_volume_net: the first half of the U-Net for encoding cost
            volume features and image prior features.
        depth_decoder: second half of the U-Net for decoding feautures into
            depth maps at multiple resolutions.
        si_loss: scale invariant loss module.
        grad_loss: multi scale gradient loss module.
        abs_loss: L1 loss module
        ms_loss_fn: type of loss to use at multiple scales
        normals_loss: module for computing normal losses
        mv_depth_loss: multi view depth loss.
        compute_normals: module for computing normals
        tensor_formatter: helper class for reshaping tensors - batching and
            unbatching views.

    A note on losses: we report more losses than we actually use for
    backprop.

    We use the term cur/current for the refernce frame (from the paper)whose
    depth we predict and src/soruce for source (neighborhood) frames used
    for matching.

    """

    def __init__(self, opts):
        """Inits a depth model.

        Args: opts: Options config object with:

        opts.image_encoder_name: the type of image encoder used for the
            image prior. Supported in this version is EfficientNet.
        opts.cv_encoder_type: the type of cost volume encoder to use. The
            only supported version here is a multi_scale encoder that takes
            in features from the cost volume and features at multiple scales
            from the encoder used for image priors.
        opts.matching_num_depth_bins: number of depth planes used for
            MVS in the cost volume.
        opts.matching_scale: w.r.t to the predicted depth map, this the
            scale at which we match in the cost volume. 0 indicates matching
            at the resolution of the depth map. 1 indicates matching at half
            that resolution.
        opts.depth_decoder_name: the type of decoder to use for decoding
            features into depth. We're using a U-Net++ like architure in
            mvsanywhere.
        opts.image_width, opts.image_height: incoming image width and
            height.
        opts.loss_type: type of loss to use at multiple scales. Final
            supported verison here is log_l1.
        opts.feature_volume_type: the type of cost volume to use. Supported
            types are simple_cost_volume for a dot product based reduction
            or an mlp_feature_volume for metadata laced feature reduction.
        opts.matching_model: type of matching model to use. 'resnet' and
            'fpn' are supported.
        opts.matching_feature_dims: number of features dimensions output
            from the matching encoder.
        opts.model_num_views: number of views to expect in each tuple of
            frames (refernece/current frame + source frames)

        """
        super().__init__()
        self.save_hyperparameters()

        self.run_opts = opts

        # iniitalize the encoder for strong image priors
        if "efficientnet" in self.run_opts.image_encoder_name:
            self.encoder = timm.create_model(
                "tf_efficientnetv2_s_in21ft1k", pretrained=True, features_only=True
            )

            self.encoder.num_ch_enc = self.encoder.feature_info.channels()
        elif "resnet18d" in self.run_opts.image_encoder_name:
            self.encoder = timm.create_model("resnet18d", pretrained=True, features_only=True)

            self.encoder.num_ch_enc = self.encoder.feature_info.channels()
        elif 'dinov2' in self.run_opts.image_encoder_name:
            intermediate_layers_idx = {
                'dinov2_vits14': [0, 2, 4, 5, 8, 11],
                'dinov2_vitb14': [0, 2, 4, 5, 8, 11], 
                'dinov2_vitl14': [4, 11, 17, 23], 
                'dinov2_vitg14': [9, 19, 29, 39]
            }[self.run_opts.image_encoder_name]
            self.encoder = DINOv2(
                self.run_opts.image_encoder_name,
                num_intermediate_layers=intermediate_layers_idx
            )
            if self.run_opts.da_weights_path is not None:
                self.encoder.load_da_weights(self.run_opts.da_weights_path)
        else:
            raise ValueError("Unrecognized option for image encoder type!")

        # iniitalize the first half of the U-Net, encoding the cost volume
        # and image prior image feautres
        if self.run_opts.cv_encoder_type == "multi_scale_encoder":
            self.cost_volume_net = CVEncoder(
                num_ch_cv=self.run_opts.matching_num_depth_bins,
                num_ch_enc=self.encoder.num_ch_enc[1:],
                num_ch_outs=[64, 128, 256, 384],
            )
            dec_num_input_ch = (
                self.encoder.num_ch_enc[:1]
                + self.cost_volume_net.num_ch_enc
            )
        elif self.run_opts.cv_encoder_type == 'vit_encoder':
            intermediate_layers_idx = {
                'dinov2_vits14': [2, 5, 8, 11],
                'dinov2_vitb14': [2, 5, 8, 11], 
                'dinov2_vitl14': [4, 11, 17, 23], 
                'dinov2_vitg14': [9, 19, 29, 39]
            }[self.run_opts.image_encoder_name]
            self.cost_volume_net = ViTCVEncoder(
                model_name=self.run_opts.image_encoder_name,
                num_ch_cv=self.run_opts.matching_num_depth_bins,
                feat_fuser_layers_idx=intermediate_layers_idx,
                intermediate_layers_idx=intermediate_layers_idx
            )
        else:
            raise ValueError("Unrecognized option for cost volume encoder type!")

        # iniitalize the final depth decoder
        if self.run_opts.depth_decoder_name == "unet_pp":
            self.depth_decoder = DepthDecoderPP(dec_num_input_ch, num_output_channels=self.run_opts.matching_num_depth_bins)
        elif self.run_opts.depth_decoder_name == "skip":
            self.depth_decoder = SkipDecoderRegression(dec_num_input_ch)
        elif self.run_opts.depth_decoder_name == "dpt" or "dpt" in self.run_opts.depth_decoder_name:
            self.depth_decoder = DPTHead(model_name=self.run_opts.image_encoder_name, prediction_scale=self.run_opts.prediction_scale)
            if self.run_opts.da_weights_path is not None:
                self.depth_decoder.load_da_weights(self.run_opts.da_weights_path)
        else:
            raise ValueError("Unrecognized option for depth decoder name!")

        # all the losses
        self.si_loss = ScaleInvariantLoss()
        self.grad_loss = MSGradientLoss()
        self.abs_loss = nn.L1Loss()
        self.normals_loss = NormalsLoss()
        self.mv_depth_loss = MVDepthLoss(
            int(self.run_opts.image_height * self.run_opts.prediction_scale),
            int(self.run_opts.image_width * self.run_opts.prediction_scale),
        )

        # Pick the multiscale loss
        if self.run_opts.loss_type == "log_l1":
            self.ms_loss_fn = self.abs_loss
        else:
            raise ValueError(f"loss_type: {self.run_opts.loss_type} unknown")

        # used for normals loss
        self.compute_normals = NormalGenerator(
            int(self.run_opts.image_height * self.run_opts.prediction_scale),
            int(self.run_opts.image_width * self.run_opts.prediction_scale),
        )

        # what type of cost volume are we using?
        if self.run_opts.feature_volume_type == "simple_cost_volume":
            cost_volume_class = CostVolumeManager
            if opts.fast_cost_volume:
                cost_volume_class = FastCostVolumeManager
        elif self.run_opts.feature_volume_type == "mlp_feature_volume":
            cost_volume_class = FeatureVolumeManager
            if opts.fast_cost_volume:
                cost_volume_class = FastFeatureVolumeManager
        elif self.run_opts.feature_volume_type == "view_agnostic_mlp_feature_volume":
            cost_volume_class = ViewAgnosticFeatureVolumeManager
            if opts.fast_cost_volume:
                cost_volume_class = FastViewAgnosticFeatureVolumeManager
        else:
            raise ValueError(
                f"Unrecognized option {self.run_opts.feature_volume_type} "
                f"for feature volume type!"
            )

        self.cost_volume = cost_volume_class(
            matching_height=int(self.run_opts.image_height * self.run_opts.matching_scale),
            matching_width=int(self.run_opts.image_width * self.run_opts.matching_scale),
            num_depth_bins=self.run_opts.matching_num_depth_bins,
            matching_dim_size=self.run_opts.matching_feature_dims,
            num_source_views=opts.model_num_views - 1,
        )

        # init the matching encoder. resnet is fast and is the default for
        # results in the paper, fpn is more accurate but much slower.
        if "resnet" == self.run_opts.matching_encoder_type:
            self.matching_model = ResnetMatchingEncoder(18, self.run_opts.matching_feature_dims)
        elif "unet_encoder" == self.run_opts.matching_encoder_type:
            self.matching_model = UNetMatchingEncoder()
        else:
            raise ValueError(
                f"Unrecognized option {self.run_opts.matching_encoder_type} "
                f"for matching encoder type!"
            )

        self.tensor_formatter = TensorFormatter()

        self.color_aug = CustomColorJitter(0.2, 0.2, 0.2, 0.2)

        self.automatic_optimization = False

    def compute_matching_feats(
        self,
        cur_image,
        src_image,
        unbatched_matching_encoder_forward,
    ):
        """
        Computes matching features for the current image (reference) and
        source images.

        Unfortunately on this PyTorch branch we've noticed that the output
        of our ResNet matching encoder is not numerically consistent when
        batching. While this doesn't affect training (the changes are too
        small), it does change and will affect test scores. To combat this
        we disable batching through this module when testing and instead
        loop through images to compute their feautures. This is stable and
        produces exact repeatable results.

        Args:
            cur_image: image tensor of shape B3HW for the reference image.
            src_image: images tensor of shape BM3HW for the source images.
            unbatched_matching_encoder_forward: disable batching and loops
                through iamges to compute feaures.
        Returns:
            matching_cur_feats: tensor of matching features of size bchw for
                the reference current image.
            matching_src_feats: tensor of matching features of size BMcHW
                for the source images.
        """
        if unbatched_matching_encoder_forward:
            all_frames_bm3hw = torch.cat([cur_image.unsqueeze(1), src_image], dim=1)
            batch_size, num_views = all_frames_bm3hw.shape[:2]
            all_frames_B3hw = tensor_bM_to_B(all_frames_bm3hw)
            matching_feats = [self.matching_model(f) for f in all_frames_B3hw.split(1, dim=0)]

            matching_feats = torch.cat(matching_feats, dim=0)
            matching_feats = tensor_B_to_bM(
                matching_feats,
                batch_size=batch_size,
                num_views=num_views,
            )

        else:
            # Compute matching features and batch them to reduce variance from
            # batchnorm when training.
            matching_feats = self.tensor_formatter(
                torch.cat([cur_image.unsqueeze(1), src_image], dim=1),
                apply_func=self.matching_model,
            )

        matching_cur_feats = matching_feats[:, 0]
        matching_src_feats = matching_feats[:, 1:].contiguous()

        return matching_cur_feats, matching_src_feats
    
    def get_heuristic_depth_range(self, cur_data, src_data):
        """
        Computes a rough depth range estimate of the current view.
        """
        cur_image = cur_data["image_b3hw"]
        src_K = src_data[f"K_matching_b44"]
        src_cam_T_world = src_data["cam_T_world_b44"]
        cur_world_T_cam = cur_data["world_T_cam_b44"]

        K_scale = 1.0 / self.run_opts.matching_scale
        
        cam_dists = (src_cam_T_world @ cur_world_T_cam.unsqueeze(1))[:, :, :3, 3].norm(dim=-1)

        baseline_min_depth = (torch.amin(cam_dists, 1) * src_K[:, 0, 0, 0]) * K_scale / (1.0 * cur_image.shape[-1])
        baseline_max_depth = (torch.amax(cam_dists, 1) * src_K[:, 0, 0, 0] * K_scale ) / (0.01 * cur_image.shape[-1])
        
        return baseline_min_depth, baseline_max_depth

    def refine_depth_range(self, cur_data, src_data, outputs):
        """
        Refines the depth range of the current view based on the predicted
        depth map.
        """
        pred_depth = outputs["depth_pred_s0_b1hw"]
        cur_image = cur_data["image_b3hw"]
        min_depth = cur_data["min_depth"]
        max_depth = cur_data["max_depth"]

        num_near_max = (pred_depth > max_depth * 0.95).flatten(1).sum(dim=1)
        num_near_min = (pred_depth < min_depth * 1.05).flatten(1).sum(dim=1)
        fraction_near_max = num_near_max / (cur_image.shape[-2] * cur_image.shape[-1])
        fraction_near_min = num_near_min / (cur_image.shape[-2] * cur_image.shape[-1])
        refine_max = fraction_near_max > 0.05
        refine_min = fraction_near_min > 0.05

        pred_depth = pred_depth.flatten(1)
        pred_min = torch.quantile(pred_depth, 0.05, dim=1) * 0.75
        pred_max = torch.quantile(pred_depth, 0.95, dim=1) * 1.25

        new_min = torch.maximum(min_depth, pred_min.view(min_depth.shape))
        new_max = torch.minimum(max_depth, pred_max.view(max_depth.shape))

        if refine_max.any().item():
            new_max[refine_max] = max_depth[refine_max] * 2.0
        if refine_min.any().item():
            new_min[refine_min] = min_depth[refine_min] * 0.5

        return new_min, new_max

    def forward(
        self,
        phase,
        cur_data,
        src_data,
        unbatched_matching_encoder_forward=False,
        return_mask=False,
        num_refinement_steps=0,
        estimate_depth_range=True,
    ):
        """
        Computes a forward pass through the depth model.

        This function is used for both training and inference.

        During training, a flip horizontal augmentation is used on images
        with a random chance of 50%. If you do plan on changing this flip,
        be careful on where its done. When we do need to flip, we only want
        it to apply through image encoders, but not through the cost volume.
        When we use a flip, we apply it to images before the matching
        encoder, flipping back when we pass those feautres through the cost
        volume, and then flip the cost volume's output so that they align
        with the current image's features (flipped when we use the image
        prior encoder) when we use our final U-Net.

        Args:
            phase: str defining phase of training. When phase is "train,"
                flip augmentation is used on images.
            cur_data: a dictionary with tensors for the current view. These
                include
                    "image_b3hw" for an RGB image,
                    "K_si_b44" intrinsics tensor for projecting points to
                        image space where i starts at 0 and goes up to the
                        maximum divisor scale,
                    "cam_T_world_b44" a camera extrinsics matrix for
                        transforming world points to camera coordinates,
                    and "world_T_cam_b44" a camera pose matrix for
                        transforming camera points to world coordinates.
            src_data: also a dictionary with elements similar to cur_data.
                All tensors here are expected to have batching shape B...
                instead of bM where M is the number of source images.
            unbatched_matching_encoder_forward: disable batching and loops
                through iamges to compute matching feaures, used for stable
                inference when testing. See compute_matching_feats for more
                information.
            return_mask: return a 2D mask from the cost volume for areas
                where there is source view information.
            num_refinement_steps: the number of num_refinement_steps steps to take. 
                Only when phase is "test".
        Returns:
            depth_outputs: a dictionary with outputs including
                "log_depth_pred_s{i}_b1hw" log depths where i is the
                resolution at which this depth map is. 0 represents the
                highest resolution depth map predicted at opts.depth_width,
                opts.depth_height,
                "log_depth_pred_s{i}_b1hw" depth maps in linear scale where
                    is the resolution at which this depth map is. 0
                    represents the highest resolution depth map predicted at
                    opts.depth_width, opts.depth_height,
                "lowest_cost_bhw" the argmax for likelihood along depth
                    planes from the cost volume, representing the best
                    matched depth plane at each spatial resolution,
                and "overall_mask_bhw" returned when return_mask is True and
                    is a 2D mask from the cost volume for areas where there
                    is source view information from the current view's point
                    of view.

        """

        # get all tensors from the batch dictioanries.
        cur_image = cur_data["image_b3hw"]
        src_image = src_data["image_b3hw"]
        src_K = src_data[f"K_matching_b44"]
        cur_invK = cur_data[f"invK_matching_b44"]
        src_cam_T_world = src_data["cam_T_world_b44"]
        src_world_T_cam = src_data["world_T_cam_b44"]

        cur_cam_T_world = cur_data["cam_T_world_b44"]
        cur_world_T_cam = cur_data["world_T_cam_b44"]

        with torch.cuda.amp.autocast(False):
            # Compute src_cam_T_cur_cam, a transformation for going from 3D
            # coords in current view coordinate frame to source view coords
            # coordinate frames.
            src_cam_T_cur_cam = src_cam_T_world @ cur_world_T_cam.unsqueeze(1)

            # Compute cur_cam_T_src_cam the opposite of src_cam_T_cur_cam. From
            # source view to current view.
            cur_cam_T_src_cam = cur_cam_T_world.unsqueeze(1) @ src_world_T_cam

        # flip transformation! Figure out if we're flipping. Should be true if
        # we are training and a coin flip says we should.
        flip_threshold = 0.5 if phase == "train" else 0.0
        flip = torch.rand(1).item() < flip_threshold

        if flip:
            # flip all images.
            cur_image = torch.flip(cur_image, (-1,))
            src_image = torch.flip(src_image, (-1,))

        # Compute image features for the current view. Used for a strong image
        # prior.
        if not "cur_feats" in cur_data:
            cur_data["cur_feats"] = self.encoder(cur_image)
        cur_feats = cur_data["cur_feats"]

        # Compute matching features
        if not "matching_cur_feats" in cur_data or not "matching_src_feats" in src_data:
            cur_data["matching_cur_feats"], src_data["matching_src_feats"] = (
                self.compute_matching_feats(
                    cur_image, src_image, unbatched_matching_encoder_forward
                )
            )
        matching_cur_feats = cur_data["matching_cur_feats"]
        matching_src_feats = src_data["matching_src_feats"]

        if flip:
            # now (carefully) flip matching features back for correct MVS.
            matching_cur_feats = torch.flip(matching_cur_feats, (-1,))
            matching_src_feats = torch.flip(matching_src_feats, (-1,))

        if not "min_depth" in cur_data or not "max_depth" in cur_data or estimate_depth_range:
            # Compute min and max depth from the current view's intrinsics and poses.
            min_depth, max_depth = self.get_heuristic_depth_range(cur_data, src_data)
        else:
            min_depth, max_depth = cur_data["min_depth"], cur_data["max_depth"]
        
        # Get min and max depth to the right shape, device and dtype
        min_depth = torch.as_tensor(min_depth, device=src_K.device, dtype=src_K.dtype).view(-1, 1, 1, 1)
        max_depth = torch.as_tensor(max_depth, device=src_K.device, dtype=src_K.dtype).view(-1, 1, 1, 1)
        cur_data["min_depth"] = min_depth
        cur_data["max_depth"] = max_depth

        # Compute the cost volume. Should be size bdhw.
        cost_volume, lowest_cost, _, overall_mask_bhw = self.cost_volume(
            cur_feats=matching_cur_feats,
            src_feats=matching_src_feats,
            src_extrinsics=src_cam_T_cur_cam,
            src_poses=cur_cam_T_src_cam,
            src_Ks=src_K,
            cur_invK=cur_invK,
            min_depth=min_depth,
            max_depth=max_depth,
            return_mask=return_mask,
        )

        if flip:
            # OK, we've computed the cost volume, now we need to flip the cost
            # volume to have it aligned with flipped image prior features
            cost_volume = torch.flip(cost_volume, (-1,))

        # Encode the cost volume and current image features
        if self.run_opts.cv_encoder_type == "multi_scale_encoder":
            cost_volume_features = self.cost_volume_net(
                cost_volume,
                cur_feats[1:],
            )
            cur_feats = cur_feats[:1] + cost_volume_features
        elif self.run_opts.cv_encoder_type == "vit_encoder":
            cur_feats = self.cost_volume_net(
                        cost_volume, 
                        cur_feats,
                    )
 
        # Decode into depth at multiple resolutions.
        if "dpt" in self.run_opts.depth_decoder_name:
            B, C, H, W = cur_image.shape
            depth_outputs = self.depth_decoder(cur_feats, H // 16, W // 16)
        else:
            depth_outputs = self.depth_decoder(cur_feats)

        # loop through depth outputs, flip them if we need to and get linear
        # scale depths.
        for k in list(depth_outputs.keys()):

            if 'sky' in k:
                continue

            log_depth = depth_outputs[k].float()
            log_depth = torch.log(min_depth) + torch.log(max_depth / min_depth) * torch.sigmoid(log_depth)

            if flip:
                # now flip the depth map back after final prediction
                log_depth = torch.flip(log_depth, (-1,))

            depth_outputs[k] = log_depth
            depth_outputs[k.replace("log_", "")] = torch.exp(log_depth)

        for k in list(depth_outputs.keys()):
            if 'sky' not in k:
                continue

            skymask = depth_outputs[k]
            if flip:
                skymask = torch.flip(skymask, (-1,))

            depth_outputs[k] = torch.sigmoid(skymask)
            depth_outputs[k.replace("sky", "depth_masked")] = (
                depth_outputs[k.replace("sky", "depth")] * (torch.sigmoid(skymask) < 0.5)
            )

        # include argmax likelihood depth estimates from cost volume and
        # overall source view mask.
        depth_outputs["lowest_cost_bhw"] = lowest_cost
        depth_outputs["overall_mask_bhw"] = overall_mask_bhw

        if num_refinement_steps > 0:
            new_min, new_max = self.refine_depth_range(cur_data, src_data, depth_outputs)
            cur_data["min_depth"] = new_min
            cur_data["max_depth"] = new_max

            return self.forward(
                phase=phase,
                cur_data=cur_data,
                src_data=src_data,
                unbatched_matching_encoder_forward=unbatched_matching_encoder_forward,
                return_mask=return_mask,
                num_refinement_steps=num_refinement_steps - 1,
                estimate_depth_range=False,
            )

        return depth_outputs

    def compute_losses(self, cur_data, src_data, outputs):
        """Compute losses for the current view's depth.

        We compute more losses than we actually use for backprop here. The
        final loss cocktail is stored in 'loss'.

        Args:
            cur_data: current view's data from the dataloader.
            src_data: source view data from the dataloader. Should also
                include GT depth for the multi view loss.
            outputs: outputs from the model, see forward for details. Should
                also include "normals_pred_b3hw" an estimate of normals from
                predicted depth.
        Returns:
            losses: a dictionary with losses for this batch. This includes:
                "loss": the final combined loss for backprop as defined in
                Equation 6 in the SimpleRecon paper,
                "si_loss": a scale invariant loss,
                "grad_loss": a multi scale gradient loss, Equation 3 in the
                    paper,
                "abs_loss": absolute difference L1 loss,
                "normals_loss": loss on estimated normals from depth,
                    Equation 4 in the paper,
                "ms_loss": multi scale regression loss, Equation 2 from the
                    paper.
                "mv_loss": multi-view depth regression loss as defined by
                    Equation 5,
                "inv_abs_loss": absolute difference on inverted depths,
                and "log_l1_loss": absolute difference on logged depths.
        """
        depth_gt = cur_data["depth_b1hw"]
        normals_gt = cur_data["normals_b3hw"]
        mask_b = cur_data["mask_b_b1hw"]
        mask = cur_data["mask_b1hw"]
        skymask = cur_data["skymask_b1hw"]
        depth_pred = outputs["depth_pred_s0_b1hw"]
        log_depth_pred = outputs["log_depth_pred_s0_b1hw"]
        normals_pred = outputs["normals_pred_b3hw"]

        log_depth_gt = torch.log(depth_gt)
        found_scale = False
        ms_loss = 0
        sky_loss = 0.0
        # Mask sparse arrays
        with torch.no_grad():
            sparsity = torch.isnan(depth_gt).sum(dim=[1, 2, 3]) / np.prod(depth_gt.shape[-2:])
            l1_loss = torch.abs(log_depth_gt - log_depth_pred)
            top_20_err = torch.stack([(
                torch.quantile(l1_loss[i][mask_b[i]], 0.8) if torch.any(mask_b[i]).item() else torch.tensor(torch.inf)).cuda() for i in range(mask_b.shape[0])
            ])
            mask_b_top20 = mask_b & (l1_loss < torch.where(sparsity > 0.5, top_20_err, torch.inf)[:, None, None, None])
            mask_b_dense = mask_b & (sparsity < 0.5)[:, None, None, None]

        for i in range(4):
            if f"log_depth_pred_s{i}_b1hw" in outputs:
                log_depth_pred_resized = F.interpolate(
                    outputs[f"log_depth_pred_s{i}_b1hw"],
                    size=depth_gt.shape[-2:],
                    mode="nearest",
                )
                ms_loss += (
                    self.ms_loss_fn(log_depth_gt[mask_b_top20], log_depth_pred_resized[mask_b_top20]) / 2**i
                )
                found_scale = True

            if f"sky_pred_s{i}_b1hw" in outputs:
                sky_pred_resized = F.interpolate(
                    outputs[f"sky_pred_s{i}_b1hw"],
                    size=skymask.shape[-2:],
                    mode="nearest",
                )
                sky_loss += torch.nan_to_num(
                    nn.functional.binary_cross_entropy_with_logits(sky_pred_resized[torch.isfinite(skymask)], skymask[torch.isfinite(skymask)].float()) / 2**i
                )
                found_scale = True

        if not found_scale:
            raise Exception("Could not find a valid scale to compute si loss!")

        abs_loss = self.abs_loss(depth_gt[mask_b], depth_pred[mask_b])
        si_loss = self.si_loss(log_depth_gt[mask_b], log_depth_pred[mask_b])

        mask_b_limit = torch.logical_and(mask_b, depth_pred > 0.1)
        inv_abs_loss = self.abs_loss(1 / depth_gt[mask_b_limit], 1 / depth_pred[mask_b_limit])

        log_l1_loss = self.abs_loss(log_depth_gt[mask_b], log_depth_pred[mask_b])
        
        depth_gt_grad = depth_gt.clone()
        depth_gt_grad[~mask_b_dense] = torch.nan
        grad_loss = torch.nan_to_num(self.grad_loss(depth_gt, depth_pred))

        normals_gt[~mask_b_dense.expand(-1, 3, -1, -1)] = torch.nan
        normals_loss = torch.nan_to_num(self.normals_loss(normals_gt, normals_pred))

        loss = ms_loss + sky_loss * 0.1 + 1.0 * grad_loss + 1.0 * normals_loss

        losses = {
            "loss": loss,
            "si_loss": si_loss,
            "grad_loss": grad_loss,
            "sky_loss": sky_loss,
            "abs_loss": abs_loss,
            "normals_loss": normals_loss,
            "ms_loss": ms_loss,
            "inv_abs_loss": inv_abs_loss,
            "log_l1_loss": log_l1_loss,
        }
        return losses

    def step(self, phase, batch, batch_idx):
        """Takes a training/validation step through the model.

        phase: "train" or "val". "train" will signal this function and
            others log results and use flip augmentation.
        batch: (cur_data, src_data) where cur_data is a dict with data on
            the current (reference) view and src_data is a dict with data on
            source views.
        """
        cur_data, src_data = batch
        batch_size = cur_data["image_b3hw"].shape[0]

        empty_cur = (cur_data["image_b3hw"].sum(dim=(1, 2, 3)) == 0).any().item()
        empty_src = (src_data["image_b3hw"].sum(dim=(2, 3, 4)) == 0).any().item()
        if empty_cur or empty_src:
            # Corrupted image within the batch, skip
            return 0.0

        if phase == "train":
            cur_data["image_b3hw"] = self.color_aug(cur_data["image_b3hw"], denormalize_first=True)
            for src_ind in range(src_data["image_b3hw"].shape[1]):
                src_data["image_b3hw"][:, src_ind] = self.color_aug(
                    src_data["image_b3hw"][:, src_ind], denormalize_first=True
                )

        # forward pass through the model.
        outputs = self(phase, cur_data, src_data, estimate_depth_range=False)

        pred_H, pred_W = outputs["depth_pred_s0_b1hw"].shape[-2:]
        if pred_H != self.compute_normals.height or pred_W != self.compute_normals.width:
            self.compute_normals = NormalGenerator(pred_H, pred_W).to(outputs["depth_pred_s0_b1hw"].device)
            self.mv_depth_loss = MVDepthLoss(pred_H, pred_W).to(outputs["depth_pred_s0_b1hw"].device)

        depth_pred = outputs["depth_pred_s0_b1hw"]
        depth_pred_lr = outputs["depth_pred_s3_b1hw"]
        skymask_pred = torch.sigmoid(outputs["sky_pred_s0_b1hw"])
        cv_min = outputs["lowest_cost_bhw"]

        depth_gt = cur_data["depth_b1hw"]
        mask = cur_data["mask_b1hw"]
        skymask_gt = cur_data["skymask_b1hw"].float()
        mask_b = cur_data["mask_b_b1hw"]

        # estimate normals for groundtruth
        normals_gt = self.compute_normals(depth_gt, cur_data["invK_s0_b44"])
        cur_data["normals_b3hw"] = normals_gt

        # estimate normals for depth
        normals_pred = self.compute_normals(depth_pred, cur_data["invK_s0_b44"])
        outputs["normals_pred_b3hw"] = normals_pred

        # compute losses
        losses = self.compute_losses(cur_data, src_data, outputs)

        if phase == "train":
            self.manual_backward(losses["loss"] / 2.0)

            if (batch_idx + 1) % 2 == 0:
                optimizer_sr, optimizer_da_enc, optimizer_da_dec = self.optimizers()
            
                optimizer_sr.step()
                optimizer_da_enc.step()
                optimizer_da_dec.step()

                optimizer_sr.zero_grad()
                optimizer_da_enc.zero_grad()
                optimizer_da_dec.zero_grad()

                scheduler_sr, scheduler_da_enc, scheduler_da_dec = self.lr_schedulers()
                scheduler_sr.step()
                scheduler_da_enc.step()
                scheduler_da_dec.step()

        #
        is_train = phase == "train"
        global_step = self.global_step // 3

        if batch_idx % 2 == 0:
            # logging and validation
            with torch.inference_mode():
                # log images for train.
                if (global_step % self.trainer.log_every_n_steps == 0 and is_train) or (not is_train and batch_idx == 0):

                    prefix = "train" if is_train else "val"
                    for i in range(min(4, batch_size)):
                        mask_i = mask[i].float().cpu()
                        depth_gt_viz_i, vmin, vmax = colormap_image(
                            depth_gt[i].float().cpu(), mask_i, return_vminvmax=True
                        )
                        depth_pred_viz_i = colormap_image(
                            depth_pred[i].float().cpu(), vmin=vmin, vmax=vmax
                        )
                        cv_min_viz_i = colormap_image(
                            cv_min[i].unsqueeze(0).float().cpu(), vmin=vmin, vmax=vmax
                        )
                        depth_pred_lr_viz_i = colormap_image(
                            depth_pred_lr[i].float().cpu(), vmin=vmin, vmax=vmax
                        )
                        skymask_gt_viz_i, vmin, vmax = colormap_image(
                            skymask_gt[i].float().cpu(), return_vminvmax=True
                        )
                        skymask_pred_viz_i = colormap_image(
                            skymask_pred[i].float().cpu(), vmin=vmin, vmax=vmax
                        )

                        image_i = reverse_imagenet_normalize(cur_data["image_b3hw"][i])

                        grid = torchvision.utils.make_grid(
                            src_data["image_b3hw"][i], nrow=3
                        )
                        grid = F.interpolate(grid[None], image_i.shape[-2:], mode="bilinear")[0]
                        grid = reverse_imagenet_normalize(grid)

                        self.logger.experiment.add_image(f"{prefix}_image/{i}", image_i, global_step)
                        self.logger.experiment.add_image(f"{prefix}_src_images/{i}", grid, global_step)
                        self.logger.experiment.add_image(
                                f"{prefix}_depth_gt/{i}", depth_gt_viz_i, global_step
                        )
                        self.logger.experiment.add_image(
                            f"{prefix}_depth_pred/{i}", depth_pred_viz_i, global_step
                        )
                        self.logger.experiment.add_image(
                            f"{prefix}_depth_pred_lr/{i}", depth_pred_lr_viz_i, global_step
                        )
                        self.logger.experiment.add_image(
                            f"{prefix}_normals_gt/{i}", 0.5 * (1 + normals_gt[i]), global_step
                        )
                        self.logger.experiment.add_image(
                            f"{prefix}_normals_pred/{i}", 0.5 * (1 + normals_pred[i]), global_step
                        )
                        self.logger.experiment.add_image(f"{prefix}_cv_min/{i}", cv_min_viz_i, global_step)

                        self.logger.experiment.add_image(
                                f"{prefix}_skymask_gt/{i}", skymask_gt_viz_i, global_step
                        )
                        self.logger.experiment.add_image(
                            f"{prefix}_skymask_pred/{i}", skymask_pred_viz_i, global_step
                        )

                    self.logger.experiment.flush()

                # log losses
                for loss_name, loss_val in losses.items():
                    self.log(
                        f"{phase}/{loss_name}",
                        loss_val,
                        sync_dist=True,
                        on_step=is_train,
                        on_epoch=not is_train,
                        prog_bar=True
                    )

                # high_res_validation: it isn't always wise to load in high
                # resolution depth maps so this is an optional flag.
                if phase == "train" or not self.run_opts.high_res_validation:
                    # compute metrics at low res depth resolution for train or if
                    # validation isn't set to use high res depth
                    metrics = compute_depth_metrics(depth_gt[mask_b], depth_pred[mask_b])
                else:
                    # if we are validating or testing, we want to upscale our
                    # predictions to full-size and compare against the GT depth map,
                    # so that metrics are comparable across resolutions
                    full_size_depth_gt = cur_data["full_res_depth_b1hw"]
                    full_size_mask_b = cur_data["full_res_mask_b_b1hw"]
                    # this should be nearest to reflect test, but keeping it for
                    # backwards comparison reasons.
                    full_size_pred = F.interpolate(
                        depth_pred,
                        full_size_depth_gt.size()[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    metrics = compute_depth_metrics(
                        full_size_depth_gt[full_size_mask_b],
                        full_size_pred[full_size_mask_b],
                    )

                for metric_name, metric_val in metrics.items():
                    self.log(
                        f"{phase}_metrics/{metric_name}",
                        metric_val,
                        sync_dist=True,
                        on_step=is_train,
                        on_epoch=not is_train,
                    )

    def training_step(self, batch, batch_idx):
        """Runs a training step."""
        return self.step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """Runs a validation step."""
        return self.step("val", batch, batch_idx)

    def configure_optimizers(self):
        """Configuring optmizers and learning rate schedules.

        By default we use a stepped learning rate schedule with steps at
        70000 and 80000.

        """
        optimizer_sr = torch.optim.AdamW(
            list(self.cost_volume.parameters()) + 
            list(self.matching_model.parameters()),
            lr=self.run_opts.lr, weight_decay=self.run_opts.wd
        )
        optimizer_da_encoder = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=self.run_opts.lr_da_encoder, weight_decay=self.run_opts.wd
        )
        optimizer_da_decoder = torch.optim.AdamW(
            list(self.depth_decoder.parameters()) + list(self.cost_volume_net.parameters()), 
            lr=self.run_opts.lr_da_decoder, weight_decay=self.run_opts.wd
        )

        def lr_lambda(step):
            if step < self.run_opts.lr_steps[0]:
                return 1
            elif step < self.run_opts.lr_steps[1]:
                return 0.1
            else:
                return 0.01

        lr_scheduler_sr = torch.optim.lr_scheduler.LambdaLR(optimizer_sr, lr_lambda)
        lr_scheduler_da_encoder = torch.optim.lr_scheduler.LinearLR(optimizer_da_encoder, start_factor=1.0, end_factor=0.001, total_iters=70000)
        lr_scheduler_da_decoder = torch.optim.lr_scheduler.LinearLR(optimizer_da_decoder, start_factor=1.0, end_factor=0.001, total_iters=70000)
        return [optimizer_sr, optimizer_da_encoder, optimizer_da_decoder], [lr_scheduler_sr, lr_scheduler_da_encoder, lr_scheduler_da_decoder]
