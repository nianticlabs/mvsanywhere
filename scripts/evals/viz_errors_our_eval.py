from pathlib import Path
import numpy as np
import torch

import trimesh

from geometryhints.utils.errors import MeshErrorVisualiser
from geometryhints.utils.volume_utils import SimpleVolume

scene_id = "scene0708_00"
gt_mesh_path = Path(
    "/mnt/nas/personal/mohameds/TransformerFusionEvalData/groundtruth/scene0708_00/mesh_gt.ply"
)
visibility_volume_path = Path(
    "/mnt/nas/personal/mohameds/scannet_test_occlusion_masks/"
)

pred_mesh_path_long_range_path = Path("/mnt/nas3/personal/faleotti/geometryhints/finerecon/official/4cm/scene0708_00.ply")
pred_mesh_path_default_path = Path("/mnt/nas3/personal/mohameds/geometry_hints/outputs/no_ablate_two_pass_3_5m/scannet/default/meshes/0.02_3.5_ours/scene0708_00.ply")

assert gt_mesh_path.exists()
assert pred_mesh_path_long_range_path.exists()
# assert pred_mesh_path_default_path.exists()

assert visibility_volume_path.exists()

gt_mesh = trimesh.load(gt_mesh_path)
visibility_volume_path = (
    Path(visibility_volume_path) / scene_id / f"{scene_id}_volume.npz"
)
visibility_volume = SimpleVolume.load(visibility_volume_path)

# long range
pred_mesh_long_range = trimesh.load(pred_mesh_path_long_range_path)
mesh_error_vis = MeshErrorVisualiser(max_val=15)
mesh_error = mesh_error_vis.forward(
    prediction=pred_mesh_long_range, target=gt_mesh, visibility_volume=visibility_volume
)

trimesh.exchange.export.export_mesh(mesh_error, "long_range_0708.ply")
print("Saved long range mesh")

# default 
pred_mesh_default = trimesh.load(pred_mesh_path_default_path)
mesh_error_vis = MeshErrorVisualiser(max_val=15)
mesh_error = mesh_error_vis.forward(
    prediction=pred_mesh_default, target=gt_mesh, visibility_volume=visibility_volume
)

trimesh.exchange.export.export_mesh(mesh_error, "default_0708.ply")
print("Saved default mesh")