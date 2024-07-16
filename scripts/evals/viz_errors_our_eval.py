from pathlib import Path
import numpy as np
import torch

import trimesh

from doubletake.utils.errors import MeshErrorVisualiser
from doubletake.utils.volume_utils import SimpleVolume

scene_id = "scene0707_00"
gt_mesh_path = Path(
    f"/mnt/nas/personal/mohameds/TransformerFusionEvalData/groundtruth/{scene_id}/mesh_gt.ply"
)
visibility_volume_path = Path(
    # "/mnt/nas/personal/mohameds/scannet_test_occlusion_masks/"
    "/mnt/nas/personal/mohameds/scannet_test_occlusion_masks/"
)

pred_mesh_path_first_path = Path(
    f"/mnt/nas3/personal/faleotti/geometryhints/finerecon/official/4cm/{scene_id}.ply"
)
pred_mesh_path_second_path = Path(
    f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/no_ablate_two_pass_3_5m/scannet/default/meshes/0.02_3.5_ours/{scene_id}.ply"
)

assert gt_mesh_path.exists()
assert pred_mesh_path_first_path.exists()
# assert pred_mesh_path_second_path.exists()

assert visibility_volume_path.exists()

gt_mesh = trimesh.load(gt_mesh_path)
visibility_volume_path = Path(visibility_volume_path) / scene_id / f"{scene_id}_volume.npz"
visibility_volume = SimpleVolume.load(visibility_volume_path)

# long range
pred_mesh_first = trimesh.load(pred_mesh_path_first_path)
mesh_error_vis = MeshErrorVisualiser(max_val=15)
mesh_error = mesh_error_vis.forward(
    prediction=pred_mesh_first, target=gt_mesh, visibility_volume=visibility_volume
)

trimesh.exchange.export.export_mesh(mesh_error, f"debug_dump/first_{scene_id}.ply")
print("Saved long range mesh")

# second
pred_mesh_second = trimesh.load(pred_mesh_path_second_path)
mesh_error_vis = MeshErrorVisualiser(max_val=15)
mesh_error = mesh_error_vis.forward(
    prediction=pred_mesh_second, target=gt_mesh, visibility_volume=visibility_volume
)

trimesh.exchange.export.export_mesh(mesh_error, f"debug_dump/second_{scene_id}.ply")
print("Saved second mesh")
