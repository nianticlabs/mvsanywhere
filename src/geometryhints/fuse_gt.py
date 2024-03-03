
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

import geometryhints.modules.cost_volume as cost_volume
import geometryhints.options as options
from geometryhints.experiment_modules.densification_model import DensificationModel
from geometryhints.experiment_modules.depth_model import DepthModel
from geometryhints.experiment_modules.depth_model_cv_hint import DepthModelCVHint
from geometryhints.tools import fusers_helper
from geometryhints.utils.dataset_utils import get_dataset
from geometryhints.utils.generic_utils import cache_model_outputs, to_gpu
from geometryhints.utils.metrics_utils import (
    ResultsAverager,
    compute_depth_metrics_batched,
)
from geometryhints.utils.model_utils import get_model_class, load_model_inference
from geometryhints.utils.visualization_utils import quick_viz_export


def main(opts):
    opts.name = "fused_gt"
    
    # get dataset
    dataset_class, scans = get_dataset(
        opts.dataset, opts.dataset_scan_split_file, opts.single_debug_scan_id
    )

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(
        opts.output_base_path, opts.name, opts.dataset, opts.frame_tuple_type
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

        mesh_output_dir = os.path.join(results_path, "meshes", mesh_output_folder_name)

        Path(mesh_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"".center(80, "#"))
        print(f" Running fusion! Using {opts.depth_fuser} ".center(80, "#"))
        print(f"Output directory:\n{mesh_output_dir} ".center(80, "#"))
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

    with torch.inference_mode():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        # loop over scans
        for scan in tqdm(scans):
            # initialize fuser if we need to fuse
            if opts.run_fusion:
                fuser = fusers_helper.get_fuser(opts, scan)

            # set up dataset with current scan
            dataset = dataset_class(
                opts.dataset_path,
                split=opts.split,
                mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
                limit_to_scan_id=scan,
                include_full_res_depth=True,
                tuple_info_file_location=opts.tuple_info_file_location,
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
                fill_depth_hints=opts.fill_depth_hints,
                depth_hint_aug=opts.depth_hint_aug,
                depth_hint_dir=opts.depth_hint_dir,
                load_empty_hints=opts.load_empty_hint,
                disable_flip=True,
                rotate_images=opts.rotate_images,
                modify_to_fov=opts.modify_to_fov,
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
                
                depth_b1hw = cur_data["full_res_depth_b1hw"].clone()
                depth_b1hw[~cur_data["full_res_mask_b_b1hw"].bool()] = -1

            
                ######################### DEPTH FUSION #########################
                if opts.run_fusion:

                    color_frame = (
                        cur_data["high_res_color_b3hw"]
                        if "high_res_color_b3hw" in cur_data
                        else cur_data["image_b3hw"]
                    )

                    fuser.fuse_frames(
                        depth_b1hw,
                        cur_data["K_full_depth_b44"],
                        cur_data["cam_T_world_b44"],
                        color_frame,
                    )

            # save the fused tsdf into a mesh file
            if opts.run_fusion:
                fuser.export_mesh(
                    os.path.join(mesh_output_dir, f"{scan.replace('/', '_')}.ply"),
                )
                fuser.save_tsdf(
                    os.path.join(mesh_output_dir, f"{scan.replace('/', '_')}_tsdf.npz"),
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
