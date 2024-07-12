import math
import os
from typing import Union

import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import torch
from PIL import Image

from doubletake.utils.generic_utils import reverse_imagenet_normalize


def colormap_image(
    image_1hw,
    mask_1hw=None,
    invalid_color=(0.0, 0, 0.0),
    flip=True,
    vmin=None,
    vmax=None,
    return_vminvmax=False,
    colormap="turbo",
):
    """
    Colormaps a one channel tensor using a matplotlib colormap.

    Args:
        image_1hw: the tensor to colomap.
        mask_1hw: an optional float mask where 1.0 donates valid pixels.
        colormap: the colormap to use. Default is turbo.
        invalid_color: the color to use for invalid pixels.
        flip: should we flip the colormap? True by default.
        vmin: if provided uses this as the minimum when normalizing the tensor.
        vmax: if provided uses this as the maximum when normalizing the tensor.
            When either of vmin or vmax are None, they are computed from the
            tensor.
        return_vminvmax: when true, returns vmin and vmax.

    Returns:
        image_cm_3hw: image of the colormapped tensor.
        vmin, vmax: returned when return_vminvmax is true.


    """
    valid_vals = image_1hw if mask_1hw is None else image_1hw[mask_1hw.bool()]
    if vmin is None:
        vmin = valid_vals.min()
    if vmax is None:
        vmax = valid_vals.max()

    cmap = torch.Tensor(plt.cm.get_cmap(colormap)(torch.linspace(0, 1, 256))[:, :3]).to(
        image_1hw.device
    )
    if flip:
        cmap = torch.flip(cmap, (0,))

    h, w = image_1hw.shape[1:]

    image_norm_1hw = (image_1hw - vmin) / (vmax - vmin)
    image_int_1hw = (torch.clamp(image_norm_1hw * 255, 0, 255)).byte().long()

    image_cm_3hw = cmap[image_int_1hw.flatten(start_dim=1)].permute([0, 2, 1]).view([-1, h, w])

    if mask_1hw is not None:
        invalid_color = torch.Tensor(invalid_color).view(3, 1, 1).to(image_1hw.device)
        image_cm_3hw = image_cm_3hw * mask_1hw + invalid_color * (1 - mask_1hw)

    if return_vminvmax:
        return image_cm_3hw, vmin, vmax
    else:
        return image_cm_3hw


def image_tensor3hw_to_numpyhw3(
    image: torch.Tensor,
    scale_to_255: bool = True,
) -> np.ndarray:
    """Transforms a torch image in format 3HW to a HW3 numpy image."""

    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Input image is not a torch.Tensor. Instead got {type(image)}")

    image = image.clone().permute(1, 2, 0)
    if scale_to_255:
        image *= 255

    image_np = np.round(image.numpy()).astype(np.uint8)

    return image_np


def tile_images(
    frame_list: list[Union[np.ndarray, torch.Tensor]],
    num_rows: int = -1,
    num_cols: int = -1,
):
    """Tiles numpy frames automatically, also allows using a specific number of
    rows and columns. All images must have the same size. HW3 for numpy images
    and 3HW for torch images.

    Args:
        frame_list (List[Union[np.ndarray, torch.Tensor]]): list of equal size
            frames to stack. Either HW3 for numpy or 3HW for torch tensors.
        num_rows (int): Optional number of rows in the tile.
            If -1, will pick automatically while respecting the optional n_cols.
            Defaults to -1.
        num_cols (int): Optional number of columns in the tile.
            If -1, will pick automatically while respecting the optional n_rows.
            Defaults to -1.

    Returns:
        tiled image array

    Raises:
        ValueError: if any of the following is true:
            - frame_list is empty
            - num_cols == 0 or num_rows == 0
            - There are any inconsistencies in shapes of the images in frame_list
            - There are not enough specified rows or cols to show all images in frame_list
    """

    if len(frame_list) == 0:
        raise ValueError("Got passed an empty list.")

    if num_cols == 0 or num_rows == 0:
        raise ValueError("Got zero rows or columns.")

    # convert to numpy if they are a tensor
    for frame_ind, frame in enumerate(frame_list):
        if isinstance(frame, torch.Tensor):
            frame_list[frame_ind] = image_tensor3hw_to_numpyhw3(frame)

        if frame_list[frame_ind].ndim > 3:
            raise ValueError(
                f"Frame at ind {frame_ind} has {frame_list[frame_ind].ndim} "
                f"dimensions which is more than what this function supports (3)."
            )

    # check all have the same size
    width = frame_list[0].shape[1]
    height = frame_list[0].shape[0]

    for frame in frame_list:
        if frame.shape[1] != width or frame.shape[0] != height:
            raise ValueError(
                f"All images must have the same size. \n"
                f"First frame has shape {frame_list[0].shape} and "
                f"then found {frame.shape} in the list."
            )

    # check if the number of rows and cols are enough
    if num_rows != -1 and num_cols != -1:
        if num_rows * num_cols < len(frame_list):
            raise ValueError(
                f"Number of requested rows, {num_rows} and columns "
                f"{num_cols} allow for a total of, {num_cols*num_rows}, "
                f"which is less than the number of frames in the list, {len(frame_list)}."
            )

    # auto selection
    if num_cols == -1 and num_rows == -1:
        # pick the number of rows and columns automatically
        # trivial cases first
        if len(frame_list) == 1:
            return frame_list[0]
        elif len(frame_list) == 2:
            return cv2.hconcat(frame_list)
        elif len(frame_list) == 5 or len(frame_list) == 6:
            num_rows = 2
            num_cols = 3
        else:
            square = math.ceil(math.sqrt(len(frame_list)))
            num_rows = square
            num_cols = square
    elif num_rows == -1:
        # find out how many rows we need.
        num_rows = math.ceil(len(frame_list) / num_cols)
    elif num_cols == -1:
        # find out how many cols we need.
        num_cols = math.ceil(len(frame_list) / num_rows)

    # start populating the tiled frame
    output_list = []
    for row_index in range(num_rows):
        column_list = []
        for column_index in range(num_cols):
            frame_index = row_index * num_cols + column_index

            if frame_index < len(frame_list):
                column_list.append(frame_list[frame_index])
            else:
                # fill with zeros if no more images exist.
                column_list.append(np.zeros_like(frame_list[0]))
        output_list.append(cv2.hconcat(column_list))

    return cv2.vconcat(output_list)


def save_viz_video_frames(frame_list, path, fps=30):
    """
    Saves a video file of numpy RGB frames in frame_list.
    """
    clip = mpy.ImageSequenceClip(frame_list, fps=fps)
    clip.write_videofile(path, verbose=False, logger=None)

    return


def quick_viz_export(
    output_path, outputs, cur_data, batch_ind, valid_mask_b, batch_size, viz_fixed_min_max=False
):
    """Helper function for quickly exporting depth maps during inference."""

    if valid_mask_b.sum() == 0:
        batch_vmin = 0.0
        batch_vmax = 5.0
    else:
        batch_vmin = cur_data["full_res_depth_b1hw"][valid_mask_b].min()
        batch_vmax = cur_data["full_res_depth_b1hw"][valid_mask_b].max()

    if batch_vmax == batch_vmin:
        batch_vmin = 0.0
        batch_vmax = 5.0

    if viz_fixed_min_max:
        batch_vmin = 0.0
        batch_vmax = 5.0

    for elem_ind in range(outputs["depth_pred_s0_b1hw"].shape[0]):
        if "frame_id_string" in cur_data:
            frame_id = cur_data["frame_id_string"][elem_ind]
        else:
            frame_id = (batch_ind * batch_size) + elem_ind
            frame_id = f"{str(frame_id):6d}"

        # check for valid depths from dataloader
        if valid_mask_b[elem_ind].sum() == 0:
            sample_vmin = 0.0
            sample_vmax = 0.0
        else:
            # these will be the same when the depth map is all ones.
            sample_vmin = cur_data["full_res_depth_b1hw"][elem_ind][valid_mask_b[elem_ind]].min()
            sample_vmax = cur_data["full_res_depth_b1hw"][elem_ind][valid_mask_b[elem_ind]].max()

        # if no meaningful gt depth in dataloader, don't viz gt and
        # set vmin/max to default
        if sample_vmax != sample_vmin:
            full_res_depth_1hw = cur_data["full_res_depth_b1hw"][elem_ind]

            full_res_depth_3hw = colormap_image(
                full_res_depth_1hw, vmin=batch_vmin, vmax=batch_vmax
            )

            full_res_depth_hw3 = np.uint8(
                full_res_depth_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255
            )
            Image.fromarray(full_res_depth_hw3).save(
                os.path.join(output_path, f"{frame_id}_gt_depth.png")
            )

        if "lowest_cost_bhw" in outputs:
            lowest_cost_3hw = colormap_image(
                outputs["lowest_cost_bhw"][elem_ind].unsqueeze(0), vmin=batch_vmin, vmax=batch_vmax
            )
            pil_image = Image.fromarray(
                np.uint8(lowest_cost_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
            )
            pil_image.save(os.path.join(output_path, f"{frame_id}_lowest_cost_pred.png"))

        depth_3hw = colormap_image(
            outputs["depth_pred_s0_b1hw"][elem_ind], vmin=batch_vmin, vmax=batch_vmax
        )
        pil_image = Image.fromarray(
            np.uint8(depth_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
        )

        pil_image.save(os.path.join(output_path, f"{frame_id}_pred_depth.png"))

        main_color_3hw = cur_data["high_res_color_b3hw"][elem_ind]
        main_color_3hw = reverse_imagenet_normalize(main_color_3hw)
        pil_image = Image.fromarray(
            np.uint8(main_color_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
        )
        pil_image.save(os.path.join(output_path, f"{frame_id}_color.png"))

        if "sampled_weights_b1hw" in outputs:
            sampled_weights_3hw = colormap_image(
                outputs["sampled_weights_b1hw"][elem_ind],
                vmin=0,
                vmax=1,
                colormap="magma",
                flip="False",
            )
            pil_image = Image.fromarray(
                np.uint8(sampled_weights_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
            )
            pil_image.save(os.path.join(output_path, f"{frame_id}_sampled_weights.png"))

        if "rendered_depth_b1hw" in outputs:
            rendered_depth_3hw = colormap_image(
                outputs["rendered_depth_b1hw"][elem_ind], vmin=batch_vmin, vmax=batch_vmax
            )
            pil_image = Image.fromarray(
                np.uint8(rendered_depth_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
            )
            pil_image.save(os.path.join(output_path, f"{frame_id}_rendered_depth.png"))

        if "cv_confidence_b1hw" in outputs:
            cv_confidence_3hw = colormap_image(
                outputs["cv_confidence_b1hw"][elem_ind],
                vmin=0,
                vmax=1,
                colormap="viridis",
                flip=False,
            )
            pil_image = Image.fromarray(
                np.uint8(cv_confidence_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
            )
            pil_image.save(os.path.join(output_path, f"{frame_id}_cv_confidence.png"))


def load_and_merge_images(frame_ids, quick_viz_directory, fps=30):
    """
    Loads images, depth maps, and cost volumes from a quick viz directory and
    merges them into a single image, then produces a video.
    """
    image_list = []
    stacked_images = []

    for frame_id in frame_ids:
        rgb_image = Image.open(os.path.join(quick_viz_directory, f"{frame_id}_color.png"))
        rgb_image = np.array(rgb_image)

        height, width = rgb_image.shape[:2]

        depth = Image.open(os.path.join(quick_viz_directory, f"{frame_id}_pred_depth.png"))
        # resize to RGB image size
        depth = np.array(depth.resize([width, height]))

        lowest_cost = Image.open(
            os.path.join(quick_viz_directory, f"{frame_id}_lowest_cost_pred.png")
        )
        # resize to RGB image size
        lowest_cost = np.array(lowest_cost.resize([width, height]))

        gt_depth = Image.open(os.path.join(quick_viz_directory, f"{frame_id}_gt_depth.png"))
        # resize to RGB image size
        gt_depth = np.array(gt_depth.resize([width, height]))

        sampled_weights = Image.open(
            os.path.join(quick_viz_directory, f"{frame_id}_sampled_weights.png")
        )
        # resize to RGB image size
        sampled_weights = np.array(sampled_weights.resize([width, height]))

        rendered_depth = Image.open(
            os.path.join(quick_viz_directory, f"{frame_id}_rendered_depth.png")
        )
        # resize to RGB image size
        rendered_depth = np.array(rendered_depth.resize([width, height]))

        merged_image = tile_images(
            [
                rgb_image,
                depth,
                lowest_cost,
                gt_depth,
                sampled_weights,
                rendered_depth,
            ]
        )
        stacked_images.append(merged_image)

    save_viz_video_frames(stacked_images, os.path.join(quick_viz_directory, "merged.mp4"), fps=fps)
