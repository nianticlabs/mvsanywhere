import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

import trimesh

from geometryhints.utils.errors import MeshErrorVisualiser, TFMeshErrorVisualiser
from geometryhints.utils.volume_utils import SimpleVolume

scene_id = "scene0708_00"
output_dir = Path("debug_dump/viz_mesh_error")
output_dir.mkdir(exist_ok=True, parents=True)

gt_mesh_path = Path(
    f"/mnt/nas/personal/mohameds/TransformerFusionEvalData/groundtruth/{scene_id}/mesh_gt.ply"
)
visibility_volume_path = Path(
    "/mnt/nas/personal/mohameds/scannet_test_occlusion_masks/"
)

tf_groundtruth_dir = "/mnt/nas/personal/mohameds/TransformerFusionEvalData/groundtruth"

pred_mesh_path_paths = [
    # f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/fused_gt/scannet/default/meshes/0.02_3.0_ours/{scene_id}.ply", 
    # f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/fused_gt/scannet/default/meshes/0.02_5.0_ours/{scene_id}.ply", 
    # f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/fused_gt/scannet/default/meshes/0.02_8.0_ours/{scene_id}.ply", 
    # f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/fused_gt/scannet/dense/meshes/0.02_3.0_open3d/{scene_id}.ply", 
    # f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/fused_gt/scannet/dense/meshes/0.02_4.0_open3d/{scene_id}.ply", 
    # f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/fused_gt/scannet/dense/meshes/0.02_5.0_open3d/{scene_id}.ply", 
    # f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/fused_gt/scannet/dense/meshes/0.02_8.0_open3d/{scene_id}.ply", 
    f"/mnt/nas3/personal/faleotti/geometryhints/finerecon/official/1cm/{scene_id}.ply",
]
pred_mesh_path_paths = [Path(p) for p in pred_mesh_path_paths]

for path_ind, path in enumerate(pred_mesh_path_paths):
    assert path.exists()
    # copy the mesh to the output directory
    trimesh.exchange.export.export_mesh(trimesh.load(path), output_dir / f"{str(path.parent).split('/')[-1].replace('.', '_')}_{path.stem}_{path_ind}_original.ply")

assert visibility_volume_path.exists()

gt_mesh = trimesh.load(gt_mesh_path)
visibility_volume_path = Path(visibility_volume_path) / scene_id / f"{scene_id}_volume.npz"
our_visibility_volume = SimpleVolume.load(visibility_volume_path)


# Load occlusion mask grid, with world2grid transform.
occlusion_mask_path = os.path.join(tf_groundtruth_dir, scene_id, "occlusion_mask.npy")
tf_occlusion_mask = np.load(occlusion_mask_path)
tf_occlusion_mask = torch.from_numpy(tf_occlusion_mask).float()
world2grid_path = os.path.join(tf_groundtruth_dir, scene_id, "world2grid.txt")
tf_world2grid = np.loadtxt(world2grid_path)
# Put data to device memory.
tf_world2grid = torch.from_numpy(tf_world2grid).float()


for pred_path_ind, pred_path in tqdm(enumerate(pred_mesh_path_paths)):
    pred_mesh = trimesh.load(pred_path)
    mesh_error_vis = MeshErrorVisualiser(max_val=20)
    mesh_error = mesh_error_vis.forward(
        prediction=pred_mesh, target=gt_mesh, visibility_volume=our_visibility_volume
    )
    trimesh.exchange.export.export_mesh(mesh_error, output_dir / f"{str(pred_path.parent).split('/')[-1].replace('.', '_')}_{pred_path.stem}_{pred_path_ind}_our_mask.ply")
    
    pred_mesh = trimesh.load(pred_path)
    mesh_error_vis = TFMeshErrorVisualiser(max_val=20)
    mesh_error = mesh_error_vis.forward(
        prediction=pred_mesh, target=gt_mesh, mask=tf_occlusion_mask, transform=tf_world2grid,
    )
    trimesh.exchange.export.export_mesh(mesh_error, output_dir / f"{str(pred_path.parent).split('/')[-1].replace('.', '_')}_{pred_path.stem}_{pred_path_ind}_tf_mask.ply")