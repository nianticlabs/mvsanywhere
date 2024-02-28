from pathlib import Path

import cv2
import numpy as np
import torch

from src.geometryhints.utils.errors import DepthErrorVisualiser


def test_depth_error_vis():
    depth_error_vis = DepthErrorVisualiser()
    gt = torch.zeros((1, 1, 256, 256)).float()
    pred = torch.linspace(0, 11, 256).float().reshape(1, 1, 1, 256).repeat(1, 1, 256, 1)
    depth_error = depth_error_vis.forward(prediction=pred, target=gt)
    assert depth_error.shape == (1, 3, 256, 256)

    # import matplotlib.pyplot as plt
    # plt.imsave("depth_error.png", depth_error.permute(0, 2, 3, 1).squeeze().numpy())


def test_depth_error_real():
    target_path = Path(
        "/mnt/nas3/shared/datasets/academic_use_only/scannet/scans/scene0700_00/sensor_data/frame-000000.depth.png"
    )
    pred_path = Path(
        "/mnt/nas3/personal/mohameds/geometry_hints/outputs/hero_model_fast/scannet/default/meshes/0.04_3.0_ours/renders/scene0700_00/rendered_depth_0.png"
    )

    assert target_path.exists()
    assert pred_path.exists()

    target = cv2.imread(str(target_path), -1).astype(np.float32) / 1000
    pred = cv2.imread(str(pred_path), -1).astype(np.float32) / 1000
    pred = cv2.resize(pred, (640, 480), interpolation=cv2.INTER_NEAREST)
    mask = target > 0

    pred_ = torch.tensor(pred).reshape(1, 1, 480, 640)
    target_ = torch.tensor(target).reshape(1, 1, 480, 640)
    mask_ = torch.tensor(mask).reshape(1, 1, 480, 640)

    depth_error_vis = DepthErrorVisualiser(max_val=target.max())
    depth_error = depth_error_vis.forward(prediction=pred_, target=target_, mask=mask_)
    assert depth_error.shape == (1, 3, 480, 640)

    # import matplotlib.pyplot as plt
    # plt.imsave("depth_error.png", depth_error.permute(0, 2, 3, 1).squeeze().numpy())
