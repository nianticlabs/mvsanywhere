import os
from pathlib import Path

import cv2 as cv2
import numpy as np
import scipy
import scipy.interpolate
import scipy.ndimage
import torch
import torch.nn.functional as F
from tqdm import tqdm

import geometryhints.modules.cost_volume as cost_volume
import geometryhints.options as options
from geometryhints.experiment_modules.depth_model import DepthModel
from geometryhints.experiment_modules.depth_model_cv_hint import DepthModelCVHint
from geometryhints.test_ablations.depth_completion import fill_missing_values
from geometryhints.tools import fusers_helper
from geometryhints.utils.dataset_utils import get_dataset
from geometryhints.utils.generic_utils import cache_model_outputs, to_gpu
from geometryhints.utils.metrics_utils import (
    ResultsAverager,
    compute_depth_metrics_batched,
)
from geometryhints.utils.visualization_utils import quick_viz_export


def main(opts):
    # get dataset
    dataset_class, scans = get_dataset(
        opts.dataset, opts.dataset_scan_split_file, opts.single_debug_scan_id
    )

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(
        opts.output_base_path, opts.name, opts.dataset, opts.frame_tuple_type
    )

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

    load_hints = True

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
            )

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
                outputs = {}

                filled_depth_maps = []
                for rendered_depth_ind, rendered_depth_1hw in enumerate(
                    cur_data["depth_hint_b1hw"]
                ):
                    # fill holes if the mesh is empty
                    if not cur_data["depth_hint_mask_b_b1hw"][rendered_depth_ind].all():
                        rendered_depth_1hw = fill_missing_values(rendered_depth_1hw)
                        filled_depth_maps.append(rendered_depth_1hw.unsqueeze(0))
                    else:
                        filled_depth_maps.append(rendered_depth_1hw.unsqueeze(0))

                outputs["depth_pred_s0_b1hw"] = torch.cat(filled_depth_maps, dim=0)

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

                ########################### Quick Viz ##########################
                if opts.dump_depth_visualization:
                    # make a dir for this scan
                    output_path = os.path.join(viz_output_dir, scan)
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    quick_viz_export(
                        output_path,
                        outputs,
                        cur_data,
                        batch_ind,
                        valid_mask_b,
                        opts.batch_size,
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
            os.path.join(scores_output_dir, f"all_scene_avg_metrics_{opts.split}.json")
        )

        print("")
        all_frame_metrics.compute_final_average()
        all_frame_metrics.pretty_print_results(print_running_metrics=False)
        all_frame_metrics.print_sheets_friendly(
            include_metrics_names=True, print_running_metrics=False
        )
        all_frame_metrics.output_json(
            os.path.join(scores_output_dir, f"all_frame_avg_metrics_{opts.split}.json")
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
