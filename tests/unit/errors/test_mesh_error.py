from pathlib import Path

import numpy as np
import torch
import trimesh

from src.geometryhints.utils.errors import MeshErrorVisualiser


def test_mesh_error_vis():
    mesh_error_vis = MeshErrorVisualiser(max_val=2)
    gt = trimesh.creation.cylinder(radius=2, height=3)
    pred = trimesh.creation.cylinder(radius=2, height=3)
    noise = np.random.uniform(low=-2, high=2, size=(len(gt.vertices), 3))
    pred.vertices += noise

    mesh_error = mesh_error_vis.forward(prediction=pred, target=gt)

    assert np.allclose(
        np.array(pred.vertices, dtype=np.float32),
        np.array(mesh_error.vertices, dtype=np.float32),
        atol=1e-7,
        rtol=1e-7,
    )
    assert np.allclose(
        np.array(pred.faces, dtype=np.float32),
        np.array(mesh_error.faces, dtype=np.float32),
        atol=1e-7,
        rtol=1e-7,
    )

    # trimesh.exchange.export.export_mesh(mesh_error, "mesh_error.ply")


def test_real_meshes():
    gt_mesh_path = Path(
        "/mnt/nas3/shared/datasets/academic_use_only/scannet/scans_test/scene0707_00/scene0707_00_vh_clean.ply"
    )
    pred_mesh_path = Path(
        "/mnt/nas3/personal/faleotti/geometryhints/finerecon/predicted-meshes/lightning_logs/official/scene0707_00.ply"
    )

    assert gt_mesh_path.exists()
    assert pred_mesh_path.exists()

    gt_mesh = trimesh.load(gt_mesh_path, force="mesh")
    pred_mesh = trimesh.load(pred_mesh_path, force="mesh")

    mesh_error_vis = MeshErrorVisualiser(max_val=15)
    mesh_error = mesh_error_vis.forward(prediction=pred_mesh, target=gt_mesh)
    assert isinstance(mesh_error, trimesh.Trimesh)
    assert len(mesh_error.vertices) == len(pred_mesh.vertices)
    # trimesh.exchange.export.export_mesh(mesh_error, "finerecon_error.ply")


def test_real_mesh_visibility():
    gt_mesh_path = Path(
        "/mnt/nas/personal/mohameds/TransformerFusionEvalData/groundtruth/scene0707_00/mesh_gt.ply"
    )
    visibility_volume_path = Path(
        "/mnt/nas/personal/mohameds/TransformerFusionEvalData/groundtruth/scene0707_00/occlusion_mask.npy"
    )
    transform_path = Path(
        "/mnt/nas/personal/mohameds/TransformerFusionEvalData/groundtruth/scene0707_00/world2grid.txt"
    )
    pred_mesh_path = Path(
        "/mnt/nas3/personal/faleotti/geometryhints/finerecon/predicted-meshes/lightning_logs/official/scene0707_00.ply"
    )

    assert gt_mesh_path.exists()
    assert pred_mesh_path.exists()
    assert visibility_volume_path.exists()
    assert transform_path.exists()

    gt_mesh = trimesh.load(gt_mesh_path)
    pred_mesh = trimesh.load(pred_mesh_path)
    visibility_volume = torch.tensor(np.load(visibility_volume_path)).float()
    transform = torch.tensor(np.loadtxt(transform_path)).float()

    mesh_error_vis = MeshErrorVisualiser(max_val=15)
    mesh_error = mesh_error_vis.forward(
        prediction=pred_mesh, target=gt_mesh, mask=visibility_volume, transform=transform
    )
    assert isinstance(mesh_error, trimesh.Trimesh)
    assert len(mesh_error.faces) != len(pred_mesh.faces)
    trimesh.exchange.export.export_mesh(mesh_error, "finerecon_error_visibility.ply")
