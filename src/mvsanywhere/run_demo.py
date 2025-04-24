"""
    Predicts depth maps using a DepthModel model. Uses an MVS dataset from
    datasets.

    All results will be stored at a base results folder (results_path) at:
        opts.output_base_path/opts.name/opts.dataset/opts.frame_tuple_type/

    frame_tuple_type is the type of image tuple used for MVS. A selection should
    be provided in the data_config file you used.

    By default will compute depth scores for each frame and provide both frame
    averaged and scene averaged metrics. The script will save these scores (per
    scene and totals) under:
        results_path/scores

    We've done our best to ensure that a torch batching bug through the matching
    encoder is fixed for (<10^-4) accurate testing by disabling image batching
    through that encoder. Run `--batch_size 4` at most if in doubt, and if
    you're looking to get as stable as possible numbers and avoid PyTorch
    gremlins, use `--batch_size 1`  for comparison evaluation.


    If you want to use this for speed, set --fast_cost_volume to True. This will
    enable batching through the matching encoder and will enable an einops
    optimized feature volume.

    This script can also be used to perform a few different auxiliary tasks,
    including:

    ### TSDF Fusion
    To run TSDF fusion provide the --run_fusion flag. You have two choices for
    fusers
    1) '--depth_fuser ours' (default) will use our fuser, whose meshes are used
        in most visualizations and for scores. This fuser does not support
        color. You must use the provided version of skimage for our custom
        measure.marching_cubes implementation that allows exporting a single
        walled mesh.
    2) '--depth_fuser open3d' will use the open3d depth fuser. This fuser
        supports color and you can enable this by using the '--fuse_color' flag.

    By default, depth maps will be clipped to 3m for fusion and a tsdf
    resolution of 0.04m3 will be used, but you can change that by changing both
    '--max_fusion_depth' and '--fusion_resolution'

    You can optionnally ask for predicted depths used for fusion to be masked
    when no vaiid MVS information exists using '--mask_pred_depths'. This is not
    enabled by default.

    You can also fuse the best guess depths from the cost volume before the
    U-Net that fuses the strong image prior. You can do this by using
    --fusion_use_raw_lowest_cost.

    Meshes will be stored under results_path/meshes/mesh_options/

    ### Cache depths
    You can optionally store depths by providing the '--cache_depths' flag.
    They will be stored at
        results_path/depths

    ### Quick viz
    There are other scripts for deeper visualizations of output depths and
    fusion, but for quick export of depth map visualization you can use
    '--dump_depth_visualization'. Visualizations will be stored at
        results_path/viz/quick_viz/


    Example command to just compute scores and cache depths
        python src/mvsanywhere/run_demo.py \
            --name mvsanywhere \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/mvsanywhere_model.yaml \
            --load_weights_from_checkpoint weights/mvsanywhere_hero.ckpt \
            --data_config_file configs/data/vdr/vdr_dense.yaml \
            --scan_parent_directory /path/to/vdr/ \
            --scan_name house \ # Scan name (house or living_room)
            --num_workers 8 \
            --batch_size 2 \
            --cache_depths

    Example command to fuse depths to get meshes
        python src/mvsanywhere/run_demo.py \
            --name mvsanywhere \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/mvsanywhere_model.yaml \
            --load_weights_from_checkpoint weights/mvsanywhere_hero.ckpt \
            --data_config_file configs/data/vdr/vdr_dense.yaml \
            --scan_parent_directory /path/to/vdr/ \
            --scan_name house \ # Scan name (house or living_room)
            --num_workers 8 \
            --batch_size 2 \
            --run_fusion

    Example command to output quick depth visualizations
        python src/mvsanywhere/run_demo.py \
            --name mvsanywhere \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/mvsanywhere_model.yaml \
            --load_weights_from_checkpoint weights/mvsanywhere_hero.ckpt \
            --data_config_file configs/data/vdr/vdr_dense.yaml \
            --scan_parent_directory /path/to/vdr/ \
            --scan_name house \ # Scan name (house or living_room)
            --num_workers 8 \
            --batch_size 2 \
            --dump_depth_visualization \

    Example command to fuse depths to get color meshes
        python src/mvsanywhere/run_demo.py \
            --name mvsanywhere \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/mvsanywhere_model.yaml \
            --load_weights_from_checkpoint weights/mvsanywhere_hero.ckpt \
            --data_config_file configs/data/vdr/vdr_dense.yaml \
            --scan_parent_directory /path/to/vdr/ \
            --scan_name house \ # Scan name (house or living_room)
            --num_workers 8 \
            --batch_size 2 \
            --run_fusion \
            --depth_fuser open3d \
            --fuse_color
"""


import datetime
import os
import random
import string
from multiprocessing import Manager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import mvsanywhere.options as options
from mvsanywhere.tools.tuple_generator import crawl_subprocess_long
from mvsanywhere.utils.dataset_utils import get_dataset
from mvsanywhere.utils.generic_utils import cache_model_outputs, to_gpu
from mvsanywhere.utils.geometry_utils import NormalGenerator
from mvsanywhere.utils.metrics_utils import (
    ResultsAverager,
    compute_depth_metrics_batched,
)
from mvsanywhere.utils.model_utils import get_model_class, load_model_inference
from mvsanywhere.utils.visualization_utils import quick_viz_export


def prepare_scan_files(opts):
    dataset_opts = opts.datasets[0]

    # get the parent path.
    parent_path = Path(opts.scan_parent_directory)
    scan_name = opts.scan_name

    # create a tmp directory for dataset files.
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    tmp_metadata_folder = Path("/tmp") / f"fmvs_{random_string}_{current_time}"

    # make a scans file and a folder for the tuples
    tmp_metadata_folder.mkdir(parents=True, exist_ok=True)
    dataset_scan_split_file = tmp_metadata_folder / "scans.txt"
    tuple_info_file_location = tmp_metadata_folder / "tuples"
    tuple_info_file_location.mkdir(parents=True, exist_ok=True)

    # check the tuple type
    frame_tuple_type = (
        "dense_offline" if dataset_opts.frame_tuple_type is None else dataset_opts.frame_tuple_type
    )

    # make a meaningful name for mv_tuple_file_suffix
    mv_tuple_file_suffix = f"_eight_view_{frame_tuple_type}.txt"

    # write the scan folder name in the scans.txt file.
    with open(dataset_scan_split_file, "w") as f:
        f.write(scan_name + "\n")

    single_debug_scan_id = scan_name

    # modify the opts file to include all the above
    dataset_opts.dataset_path = str(parent_path)
    dataset_opts.single_debug_scan_id = single_debug_scan_id
    dataset_opts.tuple_info_file_location = str(tuple_info_file_location)
    dataset_opts.dataset_scan_split_file = str(dataset_scan_split_file)
    dataset_opts.mv_tuple_file_suffix = mv_tuple_file_suffix

    opts.dataset_opts = dataset_opts

    # compute tuples
    tuples = crawl_subprocess_long(
        opts,
        single_debug_scan_id,
        0,
        Manager().Value("i", 0),
    )

    # save tuples
    with open(
        tuple_info_file_location / f"{dataset_opts.split}{mv_tuple_file_suffix}", "w"
    ) as f:
        for line in tuples:
            f.write(line + "\n")

    return dataset_opts


def init_model(opts):
    model_class_to_use = get_model_class(opts)
    model = load_model_inference(opts, model_class_to_use)
    model = model.cuda().eval()
    
    return model
            

def main(opts):
    # get dataset
    dataset_opts = prepare_scan_files(opts)
    opts.datasets[0] = dataset_opts

    assert len(opts.datasets) == 1, f"Expected only one dataset but got {len(opts.datasets)}"
    dataset_opts = opts.datasets[0]
    dataset_class, scans = get_dataset(
        dataset_opts.dataset, dataset_opts.dataset_scan_split_file, opts.single_debug_scan_id
    )

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(
        opts.output_base_path, opts.name, dataset_opts.dataset, dataset_opts.frame_tuple_type
    )

    # set up directories for fusion
    if opts.run_fusion:
        mesh_output_folder_name = (
            f"{opts.fusion_resolution}_{opts.fusion_max_depth}_{opts.depth_fuser}"
        )

        if opts.mask_pred_depth:
            mesh_output_folder_name = mesh_output_folder_name + "_masked"
        if opts.fuse_color:
            mesh_output_folder_name = mesh_output_folder_name + "_color"
        if opts.fusion_use_raw_lowest_cost:
            mesh_output_folder_name = mesh_output_folder_name + "_raw_cv"
        if opts.extended_neg_truncation:
            mesh_output_folder_name = mesh_output_folder_name + "_neg_trunc"

        mesh_output_dir = os.path.join(results_path, "meshes", mesh_output_folder_name)

        Path(mesh_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"".center(80, "#"))
        print(f" Running fusion! Using {opts.depth_fuser} ".center(80, "#"))
        print(f"Output directory:\n{mesh_output_dir} ".center(80, "#"))
        print(f"".center(80, "#"))
        print("")

    # set up directories for caching depths
    if opts.cache_depths:
        # path where we cache depth maps
        depth_output_dir = os.path.join(results_path, "depths")

        Path(depth_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"".center(80, "#"))
        print(f" Caching depths.".center(80, "#"))
        print(f"Output directory:\n{depth_output_dir} ".center(80, "#"))
        print(f"".center(80, "#"))
        print("")

    # set up directories for quick depth visualizations
    if opts.dump_depth_visualization:
        viz_output_folder_name = "quick_viz"
        viz_output_dir = os.path.join(results_path, "viz", viz_output_folder_name)

        Path(viz_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"".center(80, "#"))
        print(f" Saving quick viz.".center(80, "#"))
        print(f"Output directory:\n{viz_output_dir} ".center(80, "#"))
        print(f"".center(80, "#"))
        print("")

    # set up directory for saving scores
    scores_output_dir = os.path.join(results_path, "scores")
    Path(scores_output_dir).mkdir(parents=True, exist_ok=True)

    # Set up model. Note that we're not passing in opts as an argument, although
    # we could. We're being pretty stubborn with using the options the model had
    # used when training, saved internally as part of hparams in the checkpoint.
    # You can change this at inference by passing in 'opts=opts,' but there
    # be dragons if you're not careful.

    model = init_model(opts)

    # setting up overall result averagers
    all_frame_metrics = None
    all_scene_metrics = None

    all_frame_metrics = ResultsAverager(opts.name, f"frame metrics")
    all_scene_metrics = ResultsAverager(opts.name, f"scene metrics")

    with torch.inference_mode():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        # loop over scans
        for scan in tqdm(scans):
            # initialize fuser if we need to fuse
            if opts.run_fusion:
                from mvsanywhere.tools import fusers_helper

                fuser = fusers_helper.get_fuser(opts, scan)

            # set up dataset with current scan
            dataset = dataset_class(
                dataset_opts.dataset_path,
                split=dataset_opts.split,
                mv_tuple_file_suffix=dataset_opts.mv_tuple_file_suffix,
                limit_to_scan_id=scan,
                include_full_res_depth=True,
                tuple_info_file_location=dataset_opts.tuple_info_file_location,
                num_images_in_tuple=None,
                shuffle_tuple=opts.shuffle_tuple,
                include_high_res_color=(
                    (opts.fuse_color and opts.run_fusion) or opts.dump_depth_visualization
                ),
                include_full_depth_K=True,
                skip_frames=opts.skip_frames,
                skip_to_frame=opts.skip_to_frame,
                image_width=opts.image_width,
                image_height=opts.image_height,
                pass_frame_id=True,
                disable_flip=True,
                rotate_images=opts.rotate_images,
                matching_scale=opts.matching_scale,
                prediction_scale=opts.prediction_scale,
                prediction_num_scales=opts.prediction_num_scales,
            )

            assert len(dataset) > 0, f"Dataset {scan} is empty."

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opts.batch_size,
                shuffle=False,
                num_workers=opts.num_workers,
                drop_last=False,
            )

            # initialize scene averager
            scene_frame_metrics = ResultsAverager(opts.name, f"scene {scan} metrics")

            for batch_ind, batch in enumerate(tqdm(dataloader)):

                # get data, move to GPU
                cur_data, src_data = batch
                cur_data = to_gpu(cur_data, key_ignores=["frame_id_string"])
                src_data = to_gpu(src_data, key_ignores=["frame_id_string"])

                depth_gt = cur_data["full_res_depth_b1hw"]

                # run to get output, also measure time
                start_time.record()
                # use unbatched (looping) matching encoder image forward passes
                # for numerically stable testing. If opts.fast_cost_volume, then
                # batch.
                outputs = model(
                    phase="test",
                    cur_data=cur_data,
                    src_data=src_data,
                    unbatched_matching_encoder_forward=(not opts.fast_cost_volume),
                    return_mask=True,
                    num_refinement_steps=1,
                )

                end_time.record()
                torch.cuda.synchronize()

                elapsed_model_time = start_time.elapsed_time(end_time)

                upsampled_depth_pred_b1hw = F.interpolate(
                    outputs["depth_pred_s0_b1hw"],
                    size=(depth_gt.shape[-2], depth_gt.shape[-1]),
                    mode="nearest",
                )

                # inf max depth matches DVMVS metrics, using minimum of 0.5m
                valid_mask_b = cur_data["full_res_depth_b1hw"] > 0.5

                # Check if there any valid gt points in this sample
                if (valid_mask_b).any():
                    # compute metrics
                    metrics_b_dict = compute_depth_metrics_batched(
                        depth_gt.flatten(start_dim=1).float(),
                        upsampled_depth_pred_b1hw.flatten(start_dim=1).float(),
                        valid_mask_b.flatten(start_dim=1),
                        mult_a=True,
                    )

                    # go over batch and get metrics frame by frame to update
                    # the averagers
                    for element_index in range(depth_gt.shape[0]):
                        if (~valid_mask_b[element_index]).all():
                            # ignore if no valid gt exists
                            continue

                        element_metrics = {}
                        for key in list(metrics_b_dict.keys()):
                            element_metrics[key] = metrics_b_dict[key][element_index]

                        # get per frame time in the batch
                        element_metrics["model_time"] = elapsed_model_time / depth_gt.shape[0]

                        # both this scene and all frame averagers
                        scene_frame_metrics.update_results(element_metrics)
                        all_frame_metrics.update_results(element_metrics)

                ######################### DEPTH FUSION #########################
                if opts.run_fusion:
                    # mask predicted depths when no vaiid MVS information
                    # exists, off by default
                    if opts.mask_pred_depth:
                        overall_mask_b1hw = outputs["overall_mask_bhw"].cuda().unsqueeze(1).float()

                        overall_mask_b1hw = F.interpolate(
                            overall_mask_b1hw,
                            size=(depth_gt.shape[-2], depth_gt.shape[-1]),
                            mode="nearest",
                        ).bool()

                        upsampled_depth_pred_b1hw[~overall_mask_b1hw] = -1

                    # fuse the raw best guess depths from the cost volume, off
                    # by default
                    if opts.fusion_use_raw_lowest_cost:
                        # upsampled_depth_pred_b1hw becomes the argmax from the
                        # cost volume
                        upsampled_depth_pred_b1hw = outputs["lowest_cost_bhw"].unsqueeze(1)

                        upsampled_depth_pred_b1hw = F.interpolate(
                            upsampled_depth_pred_b1hw,
                            size=(depth_gt.shape[-2], depth_gt.shape[-1]),
                            mode="nearest",
                        )

                        overall_mask_b1hw = outputs["overall_mask_bhw"].cuda().unsqueeze(1).float()

                        overall_mask_b1hw = F.interpolate(
                            overall_mask_b1hw,
                            size=(depth_gt.shape[-2], depth_gt.shape[-1]),
                            mode="nearest",
                        ).bool()

                        upsampled_depth_pred_b1hw[~overall_mask_b1hw] = -1

                    color_frame = (
                        cur_data["high_res_color_b3hw"]
                        if "high_res_color_b3hw" in cur_data
                        else cur_data["image_b3hw"]
                    )

                    fuser.fuse_frames(
                        upsampled_depth_pred_b1hw,
                        cur_data["K_full_depth_b44"],
                        cur_data["cam_T_world_b44"],
                        color_frame,
                    )

                ########################### Quick Viz ##########################
                if opts.dump_depth_visualization:
                    # make a dir for this scan
                    output_path = os.path.join(viz_output_dir, scan)
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    if "sampled_weights_b1hw" in cur_data:
                        outputs["sampled_weights_b1hw"] = cur_data["sampled_weights_b1hw"]
                    if "rendered_depth_b1hw" in cur_data:
                        outputs["rendered_depth_b1hw"] = cur_data["rendered_depth_b1hw"]

                    quick_viz_export(
                        output_path,
                        outputs,
                        cur_data,
                        batch_ind,
                        valid_mask_b,
                        opts.batch_size,
                        opts.viz_fixed_min_max,
                    )
                ########################## Cache Depths ########################
                if opts.cache_depths:
                    output_path = os.path.join(depth_output_dir, scan)
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    cache_model_outputs(
                        output_path,
                        outputs,
                        cur_data,
                        src_data,
                        batch_ind,
                        opts.batch_size,
                    )
                ########################## Regsplatfacto ########################
                if opts.save_for_regsplatfacto:
                    output_path = os.path.join(results_path, scan, "mvsa_predictions")
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    normal_generator = NormalGenerator(
                        height=outputs["depth_pred_s0_b1hw"].shape[2],
                        width=outputs["depth_pred_s0_b1hw"].shape[3],
                        smoothing_kernel_size=0,
                        smoothing_kernel_std=0.0,
                    )
                    normals_pred_b3hw = normal_generator(
                        outputs["depth_pred_s0_b1hw"].cpu().float(),
                        cur_data["invK_s0_b44"].cpu().float(),
                    )

                    for elem_ind in range(outputs["depth_masked_pred_s0_b1hw"].shape[0]):
                        frame_id = cur_data["frame_id_string"][elem_ind]
                        frame_id = frame_id.split("/")[-1][:-4]
                        elem_filepath = os.path.join(output_path, f"{frame_id}.npz")

                        depth_1hw = outputs["depth_pred_s0_b1hw"][elem_ind].float().cpu().numpy()
                        normal_3hw = normals_pred_b3hw[elem_ind].float().cpu().numpy()

                        if opts.rotate_images:
                            # save back as landscape for splatting
                            depth_1hw = np.rot90(depth_1hw, axes=(1, 2))
                            normal_3hw = np.rot90(normal_3hw, axes=(1, 2))
                            rot_z = Rotation.from_euler("z", -90, degrees=True).as_matrix()
                            normal_3hw = np.einsum("ij,jkl->ikl", rot_z, normal_3hw)

                        np.savez(
                            elem_filepath,
                            depth=outputs["depth_pred_s0_b1hw"][elem_ind].float().cpu().numpy(),
                            normal=normals_pred_b3hw[elem_ind].float().cpu().numpy(),
                        )

            # save the fused tsdf into a mesh file
            if opts.run_fusion:
                fuser.export_mesh(
                    os.path.join(mesh_output_dir, f"{scan.replace('/', '_')}.ply"),
                )
                fuser.save_tsdf(
                    os.path.join(mesh_output_dir, f"{scan.replace('/', '_')}_tsdf.npz"),
                )

            # compute a clean average
            scene_frame_metrics.compute_final_average()

            # one scene counts as a complete unit of metrics
            all_scene_metrics.update_results(scene_frame_metrics.final_metrics)

            # print running metrics.
            print("\nScene metrics:")
            scene_frame_metrics.print_sheets_friendly(include_metrics_names=True)
            scene_frame_metrics.output_json(
                os.path.join(scores_output_dir, f"{scan.replace('/', '_')}_metrics.json")
            )
            # print running metrics.
            print("\nRunning frame metrics:")
            all_frame_metrics.print_sheets_friendly(
                include_metrics_names=False,
                print_running_metrics=True,
            )

        # compute and print final average
        print("\nFinal metrics:")
        all_scene_metrics.compute_final_average()
        all_scene_metrics.pretty_print_results(print_running_metrics=False)
        all_scene_metrics.print_sheets_friendly(
            include_metrics_names=True,
            print_running_metrics=False,
        )
        all_scene_metrics.output_json(
            os.path.join(scores_output_dir, f"all_scene_avg_metrics_{dataset_opts.split}.json")
        )

        print("")
        all_frame_metrics.compute_final_average()
        all_frame_metrics.pretty_print_results(print_running_metrics=False)
        all_frame_metrics.print_sheets_friendly(
            include_metrics_names=True, print_running_metrics=False
        )
        all_frame_metrics.output_json(
            os.path.join(scores_output_dir, f"all_frame_avg_metrics_{dataset_opts.split}.json")
        )


if __name__ == "__main__":
    # don't need grad for test.
    torch.set_grad_enabled(False)

    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    main(opts)
