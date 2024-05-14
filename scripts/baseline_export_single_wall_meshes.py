from collections import OrderedDict
from pathlib import Path
import sys, os

import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
import json
from skimage import measure
import torch
import trimesh

def to_mesh(volume, origin, voxel_size, scale_to_world=True, export_single_mesh=False):
    """Extracts a mesh from the TSDF volume using marching cubes.

    Args:
        scale_to_world: should we scale vertices from TSDF voxel coords
            to world coordinates?
        export_single_mesh: returns a single walled mesh from marching
            cubes. Requires a custom implementation of
            measure.marching_cubes that supports single_mesh

    """
    tsdf = torch.tensor(volume)
    tsdf[tsdf==1] = -1
    tsdf_np = tsdf.clamp(-1, 1).cpu().numpy()

    if export_single_mesh:
        verts, faces, norms, _ = measure.marching_cubes(
            tsdf_np,
            level=0,
            allow_degenerate=False,
            single_mesh=True,
        )
    else:
        verts, faces, norms, _ = measure.marching_cubes(
            tsdf_np,
            level=0,
            allow_degenerate=False,
        )

    if scale_to_world:
        verts = origin + verts * voxel_size

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=norms)
    
    return mesh

def main():
    #####################################################################################
    # Parse command line arguments.
    #####################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groundtruth_dir",
        action="store",
        dest="groundtruth_dir",
        default="/mnt/nas/personal/mohameds/TransformerFusionEvalData/groundtruth",
        help="Provide root directory of ground truth data",
    )
    parser.add_argument(
        "--prediction_dir",
        action="store",
        dest="prediction_dir",
        help="Provide root directory and file format of prediction data. SCAN_NAME will be replaced with the scan name.",
    )

    args = parser.parse_args()
    

    groundtruth_dir = args.groundtruth_dir
    assert os.path.exists(groundtruth_dir)
    scene_ids = sorted(os.listdir(groundtruth_dir))
    prediction_dir = args.prediction_dir

    for scene_id in tqdm(scene_ids):
        # Load predicted mesh.
        numpy_data_path = prediction_dir.replace("SCAN_NAME", scene_id)
        
        data = np.load(numpy_data_path)
        
        origin = np.array(data["origin"])
        voxel_size = data["voxel_size"]
        volume = data["tsdf"]
        
        print("Volume shape:", volume.shape)
        print("Origin:", origin)
        print("Voxel size:", voxel_size)
        
        
        mesh = to_mesh(volume, origin, voxel_size, scale_to_world=True, export_single_mesh=True)
        
        # save mesh at the same location as the numpy file
        mesh_path = numpy_data_path.replace(".npz", "_single_wall.ply")
        _ = trimesh.exchange.export.export_mesh(mesh, mesh_path)
        

if __name__ == "__main__":
    main()
