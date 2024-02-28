import functools

import numpy as np
import scipy
import torch


@functools.lru_cache(maxsize=None)
def get_mesh_grid(width, height):
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    return xx, yy


def fill_missing_values(
    rendered_depth_1hw: torch.Tensor,
):
    rendered_depth_hw = rendered_depth_1hw.squeeze().cpu().numpy()

    # from https://stackoverflow.com/questions/20753288/filling-gaps-on-an-image-using-numpy-and-scipy with modifications

    # a boolean array of (width, height) which False where there are missing values and True where there are valid (non-missing) values
    mask = ~np.isnan(rendered_depth_hw)

    # array of (number of points, 2) containing the x,y coordinates of the valid values only
    xx, yy = get_mesh_grid(rendered_depth_hw.shape[1], rendered_depth_hw.shape[0])

    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

    # the valid values in the first, second, as 1D arrays (in the same order as their coordinates in xym)
    data0 = np.ravel(rendered_depth_hw[:, :][mask])

    # interpolate
    interp0 = scipy.interpolate.LinearNDInterpolator(xym, data0)

    # reshape
    result0_hw = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

    return torch.tensor(result0_hw).unsqueeze(0).to(rendered_depth_1hw.device)
