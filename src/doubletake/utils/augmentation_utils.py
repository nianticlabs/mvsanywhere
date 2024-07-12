from typing import List, Tuple, Union

import kornia
import torch
from torch import Tensor

from doubletake.utils.generic_utils import (
    imagenet_normalize,
    reverse_imagenet_normalize,
)


class CustomColorJitter(torch.nn.Module):
    """Custom Jitter module"""

    def __init__(
        self,
        brightness: Union[Tensor, float, Tuple[float, float], List[float]],
        contrast: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        saturation: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        hue: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
    ):
        super().__init__()
        self.transform = kornia.augmentation.ColorJiggle(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            same_on_batch=False,
            p=1,
        )

    def forward(self, x: torch.Tensor, denormalize_first=False) -> torch.Tensor:
        """Apply image augmentations.
        Params:
            x: torch.Tensor in range [0,1] with shape (3,H,W) or (B,3,H,W)
            denormalize_first: denormalizes, applies coloraug, then normalizes.
        Returns:
            a tensor in range [0,1] with shape (3,H,W) or (B,3,H,W)
        """
        squeeze_dim = len(x.shape) == 3

        if denormalize_first:
            x = reverse_imagenet_normalize(x)
            x = self.transform(x, keepdim=True)
            x = imagenet_normalize(x)
        else:
            x = self.transform(x, keepdim=True)

        if squeeze_dim:
            return x.squeeze(0)
        else:
            return x
