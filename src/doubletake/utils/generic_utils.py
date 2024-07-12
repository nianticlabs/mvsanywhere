import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import kornia
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)


def copy_code_state(path):
    """Copies the code directory into the path specified using rsync. It will
    use a .gitignore file to exclude files in rsync. We preserve modification
    times in rsync."""

    # create dir
    Path(os.path.join(path)).mkdir(parents=True, exist_ok=True)

    if os.path.exists("./.gitignore"):
        # use .gitignore to remove junk
        rsync_command = f"rsync -art --exclude-from='./.gitignore' --exclude '.git' . {path}"
    else:
        print(
            "WARNING: no .gitignore found so can't use that to exlcude large "
            "files when making a back up of files in copy_code_state."
        )
        rsync_command = f"rsync -art --exclude '.git' . {path}"
    os.system(rsync_command)


def readlines(filepath):
    """Reads in a text file and returns lines in a list."""
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
    return lines


def normalize_depth_single(depth_11hw, mask_11hw, robust=False):
    if mask_11hw is not None:
        valid_depth_vals_N = depth_11hw.masked_select(mask_11hw)
    else:
        valid_depth_vals_N = torch.flatten(depth_11hw)

    num_valid_pix = valid_depth_vals_N.nelement()
    num_percentile_pix = num_valid_pix // 10

    if num_valid_pix == 0:
        return depth_11hw

    sorted_depth_vals_N = torch.sort(valid_depth_vals_N)[0]
    depth_flat_N = sorted_depth_vals_N[num_percentile_pix:-num_percentile_pix]

    if depth_flat_N.nelement() == 0:
        depth_flat_N = valid_depth_vals_N

    if robust:
        depth_shift = depth_flat_N.median()
        depth_scale = torch.mean(torch.abs(depth_flat_N - depth_shift))
    else:
        depth_shift = depth_flat_N.mean()
        depth_scale = depth_flat_N.std()

    depth_norm = (depth_11hw - depth_shift) / depth_scale

    return depth_norm


def normalize_depth(depth_b1hw: torch.Tensor, mask_b1hw: torch.Tensor = None, robust: bool = False):
    depths_11hw = torch.split(depth_b1hw, 1, 0)
    masks_11hw = [None] * len(depths_11hw) if mask_b1hw is None else torch.split(mask_b1hw, 1, 0)

    depths_norm_11hw = [
        normalize_depth_single(d, m, robust) for d, m in zip(depths_11hw, masks_11hw)
    ]

    return torch.cat(depths_norm_11hw, dim=0)


@torch.jit.script
def pyrdown(input_tensor: torch.Tensor, num_scales: int = 4):
    """Creates a downscale pyramid for the input tensor."""
    output = [input_tensor]
    for _ in range(num_scales - 1):
        down = kornia.filters.blur_pool2d(output[-1], 3)
        output.append(down)
    return output


def upsample(x):
    """
    Upsample input tensor by a factor of 2
    """
    return nn.functional.interpolate(
        x,
        scale_factor=2,
        mode="bilinear",
        align_corners=False,
    )


def batched_trace(mat_bNN):
    return mat_bNN.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


def tensor_B_to_bM(tensor_BS, batch_size, num_views):
    """Unpacks a flattened tensor of tupled elements (BS) into bMS. Tuple size
    is M."""
    # S for wild card number of dims in the middle
    # tensor_bSM = tensor_BS.unfold(0, step=num_views, size=num_views)
    # tensor_bMS = tensor_bSM.movedim(-1, 1)
    tensor_bMS = tensor_BS.view([batch_size, num_views] + list(tensor_BS.shape[1:]))

    return tensor_bMS


def tensor_bM_to_B(tensor_bMS):
    """Packs an inflated tensor of tupled elements (bMS) into BS. Tuple size
    is M."""
    # S for wild card number of dims in the middle
    num_views = tensor_bMS.shape[1]
    num_batches = tensor_bMS.shape[0]

    tensor_BS = tensor_bMS.view([num_views * num_batches] + list(tensor_bMS.shape[2:]))

    return tensor_BS


def combine_dims(x, dim_begin, dim_end):
    """Views x with the dimensions from dim_begin to dim_end folded."""
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)


def to_gpu(input_dict, key_ignores=[]):
    """ " Moves tensors in the input dict to the gpu and ignores tensors/elements
    as with keys in key_ignores.
    """
    for k, v in input_dict.items():
        if k not in key_ignores:
            input_dict[k] = v.cuda().float()
    return input_dict


def imagenet_normalize(image):
    """Normalizes an image with ImageNet statistics."""
    image = TF.normalize(tensor=image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return image


def reverse_imagenet_normalize(image):
    """Reverses ImageNet normalization in an input image."""

    image = TF.normalize(
        tensor=image,
        mean=(-2.11790393, -2.03571429, -1.80444444),
        std=(4.36681223, 4.46428571, 4.44444444),
    )
    return image


def fov_to_image_dimension(fov_degrees: float, focal_length: float) -> float:
    """Assuming the principal point is in the centre, then given an fov and a focal length
    we can work out what size the image should be
    """
    return 2.0 * focal_length * np.tan(np.deg2rad(fov_degrees / 2.0))


def crop_or_pad(image_bchw, new_height, new_width, pad_mode="constant"):
    """Crops or pads an image to a new size.
        NOTE: This function assumes that the input image is in BCHW format. and
        will round down the output size to the nearest integer.
    args:
        image_bchw: the input image in BCHW format
        new_height: the new height
        new_width: the new width
    returns:
        the cropped and/or padded image
    """

    assert image_bchw.ndim == 4, f"expected image_bchw.ndim == 4, got {image_bchw.ndim} instead."

    old_height, old_width = image_bchw.shape[2:]

    # Calculate the starting points for the crop
    top = (old_height - new_height) // 2
    left = (old_width - new_width) // 2

    # print(f"old_height: {old_height}, old_width: {old_width}")
    # print(f"new_height: {new_height}, new_width: {new_width}")
    # print(f"top: {top}, left: {left}")

    # Crop or pad the width
    if new_width <= old_width:
        # Crop the width
        image_bchw = image_bchw[:, :, :, left : left + new_width]
    else:
        # Pad the width
        pad_left = np.abs((new_width - old_width) // 2)
        image_bchw = np.pad(
            image_bchw, ((0, 0), (0, 0), (0, 0), (pad_left, pad_left)), mode=pad_mode
        )

    # Crop or pad the height
    if new_height <= old_height:
        # Crop the height
        image_bchw = image_bchw[:, :, top : top + new_height, :]
    else:
        # Pad the height
        pad_top = abs((new_height - old_height) // 2)
        image_bchw = np.pad(image_bchw, ((0, 0), (0, 0), (pad_top, pad_top), (0, 0)), mode=pad_mode)

    return image_bchw


def read_image_file(
    filepath,
    height=None,
    width=None,
    value_scale_factor=1.0,
    resampling_mode=Image.BILINEAR,
    disable_warning=False,
    target_aspect_ratio=None,
):
    """ " Reads an image file using PIL, then optionally resizes the image,
    with selective resampling, scales values, and returns the image as a
    tensor

    Args:
        filepath: path to the image.
        height, width: resolution to resize the image to. Both must not be
            None for scaling to take place.
        value_scale_factor: value to scale image values with, default is 1.0
        resampling_mode: resampling method when resizing using PIL. Default
            is PIL.Image.BILINEAR
        target_aspect_ratio: if not None, will crop the image to match this
        aspect ratio. Default is None

    Returns:
        img: tensor with (optionally) scaled and resized image data.

    """
    img = Image.open(filepath)

    if target_aspect_ratio:
        crop_image_to_target_ratio(img, target_aspect_ratio)

    # resize if both width and height are not none.
    if height is not None and width is not None:
        img_width, img_height = img.size
        # do we really need to resize? If not, skip.
        if (img_width, img_height) != (width, height):
            # warn if it doesn't make sense.
            if (width > img_width or height > img_height) and not disable_warning:
                logger.warning(
                    f"WARNING: target size ({width}, {height}) has a "
                    f"dimension larger than input size ({img_width}, "
                    f"{img_height})."
                )
            img = img.resize((width, height), resample=resampling_mode)

    img = TF.to_tensor(img).float() * value_scale_factor

    return img


def crop_image_to_target_ratio(image, target_aspect_ratio=4.0 / 3.0):
    """Crops an image to satisfy a target aspect ratio."""

    actual_aspect_ratio = image.width / image.height

    if actual_aspect_ratio > target_aspect_ratio:
        # we should crop width
        new_width = image.height * target_aspect_ratio

        left = (image.width - new_width) / 2
        top = 0
        right = (image.width + new_width) / 2
        bottom = image.height

        # Crop the center of the image
        image = image.crop((left, top, right, bottom))

    elif actual_aspect_ratio < target_aspect_ratio:
        # we should crop height
        new_height = image.width / target_aspect_ratio

        left = 0
        top = (image.height - new_height) / 2
        right = image.width
        bottom = (image.height + new_height) / 2

        # Crop the center of the image
        image = image.crop((left, top, right, bottom))

    return image


def cache_model_outputs(
    output_path,
    outputs,
    cur_data,
    src_data,
    batch_ind,
    batch_size,
):
    """Helper function for model output during inference."""

    for elem_ind in range(outputs["depth_pred_s0_b1hw"].shape[0]):
        if "frame_id_string" in cur_data:
            frame_id = cur_data["frame_id_string"][elem_ind]
        else:
            frame_id = (batch_ind * batch_size) + elem_ind
            frame_id = f"{str(frame_id):6d}"

        elem_filepath = os.path.join(output_path, f"{frame_id}.pickle")

        elem_output_dict = {}

        # for key in outputs:
        #     if outputs[key] is not None:
        #         elem_output_dict[key] = outputs[key][elem_ind].unsqueeze(0)
        #     else:
        #         elem_output_dict[key] = None

        elem_output_dict["depth_pred_s0_b1hw"] = outputs["depth_pred_s0_b1hw"][elem_ind].unsqueeze(
            0
        )
        elem_output_dict["overall_mask_bhw"] = outputs["overall_mask_bhw"][elem_ind].unsqueeze(0)

        if "cv_confidence_b1hw" in outputs:
            elem_output_dict["cv_confidence_b1hw"] = outputs["cv_confidence_b1hw"][
                elem_ind
            ].unsqueeze(0)

        # include some auxiliary information
        elem_output_dict["K_full_depth_b44"] = cur_data["K_full_depth_b44"][elem_ind].unsqueeze(0)
        elem_output_dict["K_s0_b44"] = cur_data["K_s0_b44"][elem_ind].unsqueeze(0)
        elem_output_dict["cam_T_world_b44"] = cur_data["cam_T_world_b44"][elem_ind].unsqueeze(0)

        elem_output_dict["frame_id"] = cur_data["frame_id_string"][elem_ind]
        elem_output_dict["src_ids"] = []
        for src_id_list in src_data["frame_id_string"]:
            elem_output_dict["src_ids"].append(src_id_list[elem_ind])

        with open(elem_filepath, "wb") as handle:
            pickle.dump(elem_output_dict, handle)


def get_generic_eps(tensor: Optional[Any] = None):
    """Returns the epsilon value for the given tensor.
    Args:
        tensor (Optional[Any]): The tensor to get the epsilon value for.
            Defaults to None.

    Returns:
        [float]: The epsilon value for the given tensor. If it's a floating point,
        this function will return the epsilon value for the float's dtype defined
        by PyTorch: https://pytorch.org/docs/stable/type_info.html
        The function will return 32bit epsilon if tensor is None or any other type/dtype.
    """
    if tensor is None:
        return torch.finfo(torch.float32).eps

    if not isinstance(tensor, torch.Tensor):
        return torch.finfo(torch.float32).eps

    if torch.is_floating_point(tensor):
        return torch.finfo(tensor.dtype).eps

    # If tensor is not a floating point tensor, return 32bit epsilon
    return torch.finfo(torch.float32).eps
