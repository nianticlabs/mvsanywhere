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

from doubletake.utils.volume_utils import SimpleVolume

""" 
```
CUDA_VISIBLE_DEVICES=0 python scripts/evals/mesh_eval.py \
    --groundtruth_dir SCANNET_TEST_FOLDER_PATH  \
    --prediction_dir ROOT_PRED_DIRECTORY/SCAN_NAME.ply \
    --wait_for_scan;
```

Use `--wait_for_scan` if the prediction is still being generated and you want the script to wait until a scan's mesh is available before proceeding.

Adapted from https://github.com/AljazBozic/TransformerFusion/blob/main/src/evaluation/eval.py)

"""


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
        help="Scannet test set folder root.",
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
    parser.add_argument(
        "--visibility_volume_path",
        action="store",
        help="Provide path to the visibility volume.",
        default="/mnt/nas/personal/mohameds/scannet_test_occlusion_masks/",
    )
    parser.add_argument(
        "--dont_save_scores",
        action="store_true",
        help="Don't save scores as jsons",
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
        # Load predicted mesh.
        missing_scene = False

        mesh_pred_path = prediction_dir.replace("SCAN_NAME", scene_id)

        # mesh_pred_path = os.path.join(prediction_dir, "{}.ply".format(scene_id))

        if args.wait_for_scan:
            while not os.path.exists(mesh_pred_path):
                time.sleep(30)
                print(f"Waiting for scan {scene_id} to be available in the directory")

        if not os.path.exists(mesh_pred_path):
            # We have no extracted geometry, so we use default metrics for missing scene.

            missing_scene = True

        else:
            mesh_pred = o3d.io.read_triangle_mesh(mesh_pred_path)
            if (
                np.asarray(mesh_pred.vertices).shape[0] <= 0
                or np.asarray(mesh_pred.triangles).shape[0] <= 0
            ):
                # No vertices or faces present.
                missing_scene = True

        # If no result is present for the scene, we use the maximum errors.
        if missing_scene:
            # We use default metrics for missing scene.
            print("Missing scene reconstruction: {0}".format(mesh_pred_path))
            acc_sum += max_dist
            compl_sum += max_dist
            chamfer_sum += max_dist
            prc_sum += 1.0
            rec_sum += 0.0
            f1_score_sum += 0.0

            total_num_scenes += 1
            continue

        # Load groundtruth mesh.
        mesh_gt_path = os.path.join(
            groundtruth_dir, scene_id, f"{scene_id}_vh_clean.ply".format(scene_id)
        )
        mesh_gt = o3d.io.read_triangle_mesh(mesh_gt_path)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh_gt.vertices))

        # To have a fair comparison even in the case of different mesh resolutions,
        # we always sample consistent amount of points on predicted mesh.
        pcd_pred = mesh_pred.sample_points_uniformly(number_of_points=num_points_samples, seed=0)

        # Compute gt -> predicted distance.
        # move points_gt to open3d point cloud

        dist_gt2pred = torch.tensor(gt_pcd.compute_point_cloud_distance(pcd_pred))
        dist_gt2pred = torch.minimum(dist_gt2pred, max_dist * torch.ones_like(dist_gt2pred))

        # Compute predicted -> gt distance.
        # All occluded predicted points should be masked out for , to not
        # penalize completion beyond groundtruth.
        points_pred = np.asarray(pcd_pred.points)
        points_pred = torch.from_numpy(points_pred).float().cuda()

        visibility_volume_path = (
            Path(args.visibility_volume_path) / scene_id / f"{scene_id}_volume.npz"
        )
        visibility_volume = SimpleVolume.load(visibility_volume_path)
        visibility_volume.cuda()
        vis_samples_N = visibility_volume.sample_volume(points_pred)
        valid_mask_N = vis_samples_N > 0.5

        points_pred_visible = points_pred[valid_mask_N]

        if points_pred_visible.shape[0] > 0:
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(points_pred_visible.cpu().numpy())

            dist_pred2gt = torch.tensor(pred_pcd.compute_point_cloud_distance(gt_pcd))
            dist_pred2gt = torch.minimum(dist_pred2gt, max_dist * torch.ones_like(dist_pred2gt))

        # Geometry accuracy/completion/Chamfer.
        if points_pred_visible.shape[0] > 0:
            acc = torch.mean(dist_pred2gt).item()
        else:
            acc = max_dist

        compl = torch.mean(dist_gt2pred).item()
        chamfer = 0.5 * (acc + compl)

        # Precision/recall/F1 score.
        if points_pred_visible.shape[0] > 0:
            prc = (dist_pred2gt <= dist_threshold).float().mean().item()
        else:
            prc = 0.0

        rec = (dist_gt2pred <= dist_threshold).float().mean().item()

        if prc + rec > 0:
            f1_score = 2 * prc * rec / (prc + rec)
        else:
            f1_score = 0.0

        # Update total metrics.
        acc_sum += acc
        compl_sum += compl
        chamfer_sum += chamfer
        prc_sum += prc
        rec_sum += rec
        f1_score_sum += f1_score

        total_num_scenes += 1

        scores_dict = {
            "scene_id": scene_id,
            "acc": acc,
            "compl": compl,
            "chamfer": chamfer,
            "prc": prc,
            "rec": rec,
            "f1_score": f1_score,
        }
        # Update scene stats.
        scene_scores[scene_id] = scores_dict
        scores_save_path = mesh_pred_path.split(".ply")[0] + "_scores_our_masks.json"
        # save json file

        if not args.dont_save_scores:
            with open(scores_save_path, "w") as f:
                json.dump(scores_dict, f, indent=4)

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

    # save json file
    scores_save_path = os.path.join(prediction_dir.strip("SCAN_NAME.ply"), "scores_our_masks.json")
    if not args.dont_save_scores:
        with open(scores_save_path, "w") as f:
            json.dump(scene_scores, f, indent=4)


if __name__ == "__main__":
    main()
