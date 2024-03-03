import numpy as np


def find_image_bounding_box(image: np.ndarray) -> tuple[int, int, int, int]:
    """Finds the bounding box of the content in the image, where non-content regions are white.

    Args:
        image (np.ndarray): image to find bounding box of, assumed in np.uint8 format

    Returns:
        tuple[int, int, int, int]: left, top, bottom, right coordinates of the bounding box
    """
    assert image.dtype == np.uint8

    fg_mask = ~(image == 255).all(2)

    left, top = 0, 0
    bottom, right = fg_mask.shape

    # find the top
    for row in fg_mask:
        if np.any(row):
            break
        top += 1

    # find the bottom
    for row in fg_mask[::-1]:
        if np.any(row):
            break
        bottom -= 1

    # find the left
    for col in fg_mask.T:
        if np.any(col):
            break
        left += 1

    # find the right
    for col in fg_mask.T[::-1]:
        if np.any(col):
            break
        right -= 1

    return left, top, bottom, right


def find_image_collection_bounding_box(images: list[np.ndarray]) -> tuple[int, int, int, int]:
    """Finds the tightest single bounding box for a set of images."""

    # find the minimum bounding box which avoids cropping any content from any image
    # initialise the bounding box to the size of the first image
    top, left, _ = images[0].shape
    bottom, right = 0, 0

    for image in images:
        _left, _top, _bottom, _right = find_image_bounding_box(image)
        left = min(left, _left)
        top = min(top, _top)
        right = max(right, _right)
        bottom = max(bottom, _bottom)

    assert bottom > top
    assert right > left

    return left, top, bottom, right


def tightly_crop_images(images: list[np.ndarray]):
    """Finds the tightest single bounding box for a set of images and crops them all to that size."""

    # assert all images the same size
    for image in images:
        assert image.shape == images[0].shape

    left, top, bottom, right = find_image_collection_bounding_box(images)

    # crop all the images to the minimum bounding box
    return [im[top:bottom, left:right] for im in images]
