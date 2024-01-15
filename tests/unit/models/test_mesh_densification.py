import pytest
import torch

from geometryhints.experiment_modules.densification_model import DensificationModel
from geometryhints.options import Options


@pytest.mark.parametrize("batch", (1, 2))
@pytest.mark.parametrize("views", (2, 4))
@pytest.mark.parametrize("return_mask", (True, False))
def test_depth_model_cv_hint_mlp_feature_volume(batch: int, views: int, return_mask: bool):
    opts = Options()

    opts.image_width = 512
    opts.image_height = 384

    opts.image_encoder_name = "efficientnet"
    opts.cv_encoder_type = "multi_scale_encoder"
    opts.feature_volume_type = "mlp_mesh_hint_feature_volume"
    opts.matching_feature_dims = 16
    opts.matching_encoder_type = "resnet"
    opts.depth_decoder_name = "unet_pp"
    opts.min_matching_depth = 0.25
    opts.max_matching_depth = 100
    opts.matching_num_depth_bins = 64
    opts.matching_scale = 1
    opts.loss_type = "log_l1"
    opts.model_num_views = views + 1

    model = DensificationModel(opts=opts)

    rand_mask = torch.rand((batch, 1, 384, 512)) > 0.5

    cur_data = {
        "image_b3hw": torch.rand((batch, 3, 384, 512)),
        "K_s1_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
        "invK_s1_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
        "cam_T_world_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
        "world_T_cam_b44": torch.eye(4)[None, :, :].repeat(batch, 1, 1),
        "depth_hint_b1hw": torch.rand((batch, 1, 384, 512)),
        "depth_hint_mask_b1hw": rand_mask.float(),
        "depth_hint_mask_b_b1hw": rand_mask,
    }
    src_data = {}

    prediction = model.forward(
        phase="test", cur_data=cur_data, src_data=src_data, return_mask=return_mask
    )
    assert prediction["lowest_cost_bhw"].shape == (batch, 96, 128)
    if return_mask:
        assert prediction["overall_mask_bhw"].shape == (batch, 96, 128)
    else:
        assert prediction["overall_mask_bhw"] is None
    assert prediction["log_depth_pred_s1_b1hw"].shape == (batch, 1, 96, 128)
