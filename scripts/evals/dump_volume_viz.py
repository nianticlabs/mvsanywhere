from collections import OrderedDict
import sys, os

import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
import json

""" 
```
CUDA_VISIBLE_DEVICES=1 python scripts/evals/tf_mesh_eval.py \
    --groundtruth_dir /mnt/nas/personal/mohameds/TransformerFusionEvalData/groundtruth  \
    --prediction_dir /mnt/nas3/personal/mohameds/geometry_hints/outputs/debug_eval/scannet/default/meshes/0.04_3.0_ours/SCAN_NAME.ply \
    --wait_for_scan;
```

Use `--wait_for_scan` if the prediction is still being generated and you want the script to wait until a scan's mesh is available before proceeding.

"""


def visualize_occlusion_mask(occlusion_mask, world2grid):
    # occlusion_mask = occlusion_mask[::4, ::4, ::4]

    dim_x = occlusion_mask.shape[0]
    dim_y = occlusion_mask.shape[1]
    dim_z = occlusion_mask.shape[2]

    # Generate voxel indices.
    x = torch.arange(dim_x, dtype=occlusion_mask.dtype, device=occlusion_mask.device)
    y = torch.arange(dim_y, dtype=occlusion_mask.dtype, device=occlusion_mask.device)
    z = torch.arange(dim_z, dtype=occlusion_mask.dtype, device=occlusion_mask.device)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    grid_xyz = torch.cat(
        [
            grid_x.view(dim_x, dim_y, dim_z, 1),
            grid_y.view(dim_x, dim_y, dim_z, 1),
            grid_z.view(dim_x, dim_y, dim_z, 1),
        ],
        dim=3,
    )
    print(grid_xyz.shape)
    # Filter visible points.
    grid_xyz = grid_xyz[occlusion_mask < 0.5]
    num_occluded_voxels = grid_xyz.shape[0]

    # Transform voxels to world space.
    grid2world = torch.inverse(world2grid)
    R_grid2world = grid2world[:3, :3].view(1, 3, 3).expand(num_occluded_voxels, -1, -1)
    t_grid2world = grid2world[:3, 3].view(1, 3, 1).expand(num_occluded_voxels, -1, -1)

    grid_xyz_world = (torch.matmul(R_grid2world, grid_xyz.view(-1, 3, 1)) + t_grid2world).view(
        -1, 3
    )

    return grid_xyz_world


def filter_occluded_points(points_pred, world2grid, occlusion_mask):
    dim_x = occlusion_mask.shape[0]
    dim_y = occlusion_mask.shape[1]
    dim_z = occlusion_mask.shape[2]
    num_points_pred = points_pred.shape[0]

    # Transform points to bbox space.
    R_world2grid = world2grid[:3, :3].view(1, 3, 3).expand(num_points_pred, -1, -1)
    t_world2grid = world2grid[:3, 3].view(1, 3, 1).expand(num_points_pred, -1, -1)

    points_pred_coords = (
        torch.matmul(R_world2grid, points_pred.view(num_points_pred, 3, 1)) + t_world2grid
    ).view(num_points_pred, 3)

    # Normalize to [-1, 1]^3 space.
    # The world2grid transforms world positions to voxel centers, so we need to
    # use "align_corners=True".
    points_pred_coords[:, 0] /= dim_x - 1
    points_pred_coords[:, 1] /= dim_y - 1
    points_pred_coords[:, 2] /= dim_z - 1
    points_pred_coords = points_pred_coords * 2 - 1

    # Trilinearly interpolate occlusion mask.
    # Occlusion mask is given as (x, y, z) storage, but the grid_sample method
    # expects (c, z, y, x) storage.
    visibility_mask = 1 - occlusion_mask.view(dim_x, dim_y, dim_z)
    visibility_mask = visibility_mask.permute(2, 1, 0).contiguous()
    visibility_mask = visibility_mask.view(1, 1, dim_z, dim_y, dim_x)

    points_pred_coords = points_pred_coords.view(1, 1, 1, num_points_pred, 3)

    points_pred_visibility = torch.nn.functional.grid_sample(
        visibility_mask,
        points_pred_coords.cpu(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).cuda()

    points_pred_visibility = points_pred_visibility.view(num_points_pred)

    eps = 1e-5
    points_pred_visibility = points_pred_visibility >= 1 - eps

    # Filter occluded predicted points.
    if points_pred_visibility.sum() == 0:
        # If no points are visible, we keep the original points, otherwise
        # we would penalize the sample as if nothing is predicted.
        print("All points occluded, keeping all predicted points!")
        points_pred_visible = points_pred.clone()
    else:
        points_pred_visible = points_pred[points_pred_visibility]

    return points_pred_visible


def main():
    #####################################################################################
    # Settings.
    #####################################################################################
    dist_threshold = 0.05
    max_dist = 1.0
    num_points_samples = 200000

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
    parser.add_argument(
        "--single_scene", type=str, default=None, help="Optional flag to eval only one scan."
    )
    parser.add_argument(
        "--wait_for_scan",
        action="store_true",
        help="Wait for scan to be available in the directory",
    )

    args = parser.parse_args()

    groundtruth_dir = args.groundtruth_dir
    prediction_dir = args.prediction_dir
    assert os.path.exists(groundtruth_dir)

    #####################################################################################
    # Evaluate every scene.
    #####################################################################################
    # Metrics
    acc_sum = 0.0
    compl_sum = 0.0
    chamfer_sum = 0.0
    prc_sum = 0.0
    rec_sum = 0.0
    f1_score_sum = 0.0

    total_num_scenes = 0
    scene_scores = OrderedDict()

    scene_ids = sorted(os.listdir(groundtruth_dir))
    print(args.single_scene)
    if args.single_scene is not None:
        scene_ids = [args.single_scene]

    for scene_id in tqdm(scene_ids):
        # Load groundtruth mesh.
        mesh_gt_path = os.path.join(groundtruth_dir, scene_id, "mesh_gt.ply".format(scene_id))
        mesh_gt = o3d.io.read_triangle_mesh(mesh_gt_path)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh_gt.vertices))

        # Load occlusion mask grid, with world2grid transform.
        occlusion_mask_path = os.path.join(groundtruth_dir, scene_id, "occlusion_mask.npy")
        occlusion_mask = np.load(occlusion_mask_path)

        world2grid_path = os.path.join(groundtruth_dir, scene_id, "world2grid.txt")
        world2grid = np.loadtxt(world2grid_path)

        # Put data to device memory.

        world2grid = torch.from_numpy(world2grid).float().cuda()

        # We keep occlusion mask on host memory, since it can be very large for big scenes.
        occlusion_mask = torch.from_numpy(occlusion_mask).float().cuda()
        print(occlusion_mask.shape)

        # Just for debugging: Visualize occluded points.
        occluded_pcd = o3d.geometry.PointCloud()
        occluded_pcd.points = o3d.utility.Vector3dVector(
            visualize_occlusion_mask(occlusion_mask, world2grid).cpu().numpy()
        )
        occluded_pcd.paint_uniform_color([0.7, 0.0, 0.0])
        occluded_pcd = occluded_pcd.uniform_down_sample(100)

        # save the visibility point cloud
        occluded_pcd_path = os.path.join(
            prediction_dir.strip("SCAN_NAME.ply"), f"{scene_id}_occluded.ply"
        )
        o3d.io.write_point_cloud(occluded_pcd_path, occluded_pcd)

    #####################################################################################
    # Report evaluation results.
    #####################################################################################
    # Report independent scene stats.
    # Sort by speficied metric.
    elem_ids_f_scores = [
        [scene_id, scene_scores[scene_id]["f1_score"]] for scene_id in scene_scores.keys()
    ]
    sorted_idxs = [i[0] for i in sorted(elem_ids_f_scores, key=lambda x: -x[1])]

    print()
    print("#" * 50)
    print("SCENE STATS")
    print("#" * 50)
    print()

    num_best_scenes = 20

    for i, idx in enumerate(sorted_idxs):
        if i >= num_best_scenes:
            break

        print(
            "Scene {0}: acc = {1}, compl = {2}, chamfer = {3}, prc = {4}, rec = {5}, f1_score = {6}".format(
                scene_scores[idx]["scene_id"],
                scene_scores[idx]["acc"],
                scene_scores[idx]["compl"],
                scene_scores[idx]["chamfer"],
                scene_scores[idx]["prc"],
                scene_scores[idx]["rec"],
                scene_scores[idx]["f1_score"],
            )
        )

    # Metrics summary.
    mean_acc = acc_sum / total_num_scenes
    mean_compl = compl_sum / total_num_scenes
    mean_chamfer = chamfer_sum / total_num_scenes
    mean_prc = prc_sum / total_num_scenes
    mean_rec = rec_sum / total_num_scenes
    mean_f1_score = f1_score_sum / total_num_scenes

    metrics = {
        "acc": mean_acc,
        "compl": mean_compl,
        "chamfer": mean_chamfer,
        "prc": mean_prc,
        "rec": mean_rec,
        "f1_score": mean_f1_score,
    }

    scene_scores["overall"] = metrics

    # save json file
    scores_save_path = os.path.join(prediction_dir.strip("SCAN_NAME.ply"), "scores.json")
    with open(scores_save_path, "w") as f:
        json.dump(scene_scores, f, indent=4)

    print()
    print("#" * 50)
    print("EVALUATION SUMMARY")
    print("#" * 50)
    print("{:<30} {}".format("GEOMETRY ACCURACY:", metrics["acc"]))
    print("{:<30} {}".format("GEOMETRY COMPLETION:", metrics["compl"]))
    print("{:<30} {}".format("CHAMFER:", metrics["chamfer"]))
    print("{:<30} {}".format("PRECISION:", metrics["prc"]))
    print("{:<30} {}".format("RECALL:", metrics["rec"]))
    print("{:<30} {}".format("F1_SCORE:", metrics["f1_score"]))

    print(args.prediction_dir)
    print(
        f"{metrics['acc']*100:.4f}",
        f"{(metrics['compl']*100):.4f}",
        f"{(metrics['chamfer']*100):.4f}",
        f"{(metrics['prc']):.4f}",
        f"{(metrics['rec']):.4f}",
        f"{(metrics['f1_score']):.4f}",
    )


if __name__ == "__main__":
    main()
