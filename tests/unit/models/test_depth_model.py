import pytest
import torch

from geometryhints.experiment_modules.depth_model import DepthModel
from geometryhints.options import Options


@pytest.mark.parametrize("batch", (1, 2))
@pytest.mark.parametrize("views", (2, 4))
@pytest.mark.parametrize("return_mask", (True, False))
def test_depth_model_test_simple_cost_volume(batch: int, views: int, return_mask: bool):
    opts = Options()
    opts.image_width = 512
    opts.image_height = 384

    opts.image_encoder_name = "efficientnet"
    opts.cv_encoder_type = "multi_scale_encoder"
    opts.feature_volume_type = "simple_cost_volume"
    opts.matching_feature_dims = 16
    opts.matching_encoder_type = "resnet"
    opts.depth_decoder_name = "unet_pp"
    opts.min_matching_depth = 0.25
    opts.max_matching_depth = 100
    opts.matching_num_depth_bins = 64
    opts.matching_scale = 1
    opts.loss_type = "log_l1"
    opts.model_num_views = views

    model = DepthModel(opts=opts)

    cur_data = {
        "image_b3hw": torch.rand((batch, 3, 384, 512)),
        "K_s1_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
        "invK_s1_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
        "cam_T_world_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
        "world_T_cam_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
    }
    src_data = {
        "image_b3hw": torch.rand((batch, views, 3, 384, 512)),
        "K_s1_b44": torch.eye(4)[None, None, :, :].repeat(batch, views, 1, 1),
        "invK_s1_b44": torch.eye(4)[None, None, :, :].repeat(batch, views, 1, 1),
        "cam_T_world_b44": torch.eye(4)[None, None, :, :].repeat(batch, views, 1, 1),
        "world_T_cam_b44": torch.eye(4)[None, None, :, :].repeat(batch, views, 1, 1),
    }
    prediction = model.forward(
        phase="test", cur_data=cur_data, src_data=src_data, return_mask=return_mask
    )
    assert prediction["lowest_cost_bhw"].shape == (batch, 96, 128)
    assert prediction["overall_mask_bhw"] is None
    assert prediction["log_depth_pred_s1_b1hw"].shape == (batch, 1, 96, 128)


@pytest.mark.parametrize("batch", (1, 2))
@pytest.mark.parametrize("views", (2, 4))
@pytest.mark.parametrize("return_mask", (True, False))
def test_depth_model_test_mlp_feature_volume(batch: int, views: int, return_mask: bool):
    opts = Options()

    opts.image_width = 512
    opts.image_height = 384

    opts.image_encoder_name = "efficientnet"
    opts.cv_encoder_type = "multi_scale_encoder"
    opts.feature_volume_type = "mlp_feature_volume"
    opts.matching_feature_dims = 16
    opts.matching_encoder_type = "resnet"
    opts.depth_decoder_name = "unet_pp"
    opts.min_matching_depth = 0.25
    opts.max_matching_depth = 100
    opts.matching_num_depth_bins = 64
    opts.matching_scale = 1
    opts.loss_type = "log_l1"
    opts.model_num_views = views + 1

    model = DepthModel(opts=opts)

    cur_data = {
        "image_b3hw": torch.rand((batch, 3, 384, 512)),
        "K_s1_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
        "invK_s1_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
        "cam_T_world_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
        "world_T_cam_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
    }
    src_data = {
        "image_b3hw": torch.rand((batch, views, 3, 384, 512)),
        "K_s1_b44": torch.eye(4)[None, None, :, :].repeat(batch, views, 1, 1),
        "invK_s1_b44": torch.eye(4)[None, None, :, :].repeat(batch, views, 1, 1),
        "cam_T_world_b44": torch.eye(4)[None, None, :, :].repeat(batch, views, 1, 1),
        "world_T_cam_b44": torch.eye(4)[None, None, :, :].repeat(batch, views, 1, 1),
    }
    prediction = model.forward(
        phase="test", cur_data=cur_data, src_data=src_data, return_mask=return_mask
    )
    assert prediction["lowest_cost_bhw"].shape == (batch, 96, 128)
    if return_mask:
        assert prediction["overall_mask_bhw"].shape == (batch, 96, 128)
    else:
        assert prediction["overall_mask_bhw"] is None
    assert prediction["log_depth_pred_s1_b1hw"].shape == (batch, 1, 96, 128)
