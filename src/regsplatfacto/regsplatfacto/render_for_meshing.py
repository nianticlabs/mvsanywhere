# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
This file is used to render all images in the dataset, ready for meshing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import tyro
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import Dataset, InputDataset
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.scripts.render import BaseRender, _disable_datamanager_setup
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from rich import box, style
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from typing_extensions import Annotated


def compute_mapping_from_nerfstudio_to_world(
    dataparser_outputs: DataparserOutputs,
) -> Tuple[np.ndarray, float]:
    """
    Compute the transform and scale required to map outputs from NerfStudio to world.

    :param dataparser_outputs:
    :return: 4x4 transform matrix
    : return: scale, float
    """
    assert dataparser_outputs.dataparser_transform.shape == (3, 4)
    transform_h = torch.eye(4)
    transform_h[:3, :4] = dataparser_outputs.dataparser_transform
    return torch.linalg.inv(transform_h).numpy(), 1.0 / dataparser_outputs.dataparser_scale


@dataclass
class DatasetRenderMeshing(BaseRender):
    """Render all images in the dataset for meshing

    Most of this is taken from Nerfstudio DatasetRender with minor modifications.
    The main change is that we save the outputs in an npz file, to enable them to be loaded
    and used for meshing. The main code we have added is marked with "### Regsplatfacto code ###".
    """

    output_path: Path = Path("renders")
    """Where to save the renderings"""
    data: Optional[Path] = None
    """Override path to the dataset."""
    downscale_factor: Optional[float] = None  # type: ignore
    """Scaling factor to apply to the camera image resolution.
    Unlike in the base class, here we are allowing this to be None (hence the type: ignore)"""
    split: Literal["train", "val", "test", "train+test"] = "train"
    """Split to render."""
    rescale_to_world: Optional[bool] = False
    """Whether to rescale camera poses and depth maps to the initial world scale. NerfStudio rescales the camera poses, so the trained Splat model is not in the same scale as the original data."""

    def main(self) -> None:
        config: TrainerConfig

        def update_config(config: TrainerConfig) -> TrainerConfig:
            data_manager_config = config.pipeline.datamanager
            assert isinstance(
                data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig)
            )
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            if isinstance(data_manager_config, VanillaDataManagerConfig):
                data_manager_config.train_num_images_to_sample_from = -1
                data_manager_config.train_num_times_to_repeat_images = -1
            if self.data is not None:
                data_manager_config.data = self.data
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config, "dataparser")
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(
            data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig)
        )

        for split in self.split.split("+"):
            datamanager: VanillaDataManager
            dataset: InputDataset
            if split == "train":
                with _disable_datamanager_setup(
                    data_manager_config._target
                ):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(
                        test_mode="test", device=pipeline.device
                    )

                dataset = datamanager.train_dataset
                dataparser_outputs = getattr(
                    dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs
                )
            else:
                with _disable_datamanager_setup(
                    data_manager_config._target
                ):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(
                        split=datamanager.test_split
                    )
            dataloader = FixedIndicesEvalDataloader(
                input_dataset=dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))

            (
                nerfstudio_to_world_transform,
                nerfstudio_to_world_scale,
            ) = compute_mapping_from_nerfstudio_to_world(dataparser_outputs)

            with Progress(
                TextColumn(f":movie_camera: Rendering split {split} :movie_camera:"),
                BarColumn(),
                TaskProgressColumn(
                    text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                    show_speed=True,
                ),
                ItersPerSecColumn(suffix="fps"),
                TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                TimeElapsedColumn(),
            ) as progress:
                for camera_idx, (camera, batch) in enumerate(
                    progress.track(dataloader, total=len(dataset))
                ):
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs(camera)

                        # Get the original filename
                        image_name = dataparser_outputs.image_filenames[camera_idx].relative_to(
                            images_root
                        )

                        output_path = self.output_path / split / image_name
                        output_path.parent.mkdir(exist_ok=True, parents=True)

                        ### Regsplatfacto code ###
                        # save npz containing everything we need for meshing
                        npz_path = output_path.with_suffix(".npz")

                        to_save = {"image_hw3": outputs["rgb"].cpu().numpy()}

                        depth_hw = outputs["depth"].squeeze(-1).cpu().numpy()
                        alpha_hw = outputs["accumulation"].squeeze(-1).cpu().numpy()

                        # mask out depths with zero visibilty
                        depth_hw[alpha_hw == 0.0] = 0.0

                        if self.rescale_to_world:
                            depth_hw *= nerfstudio_to_world_scale

                        to_save["depth_hw"] = depth_hw

                        if "render_clusters_class" in outputs:
                            to_save["clusters"] = outputs["render_clusters_class"].cpu().numpy()

                        cam2world_44 = np.eye(4)
                        K_44 = np.eye(4)
                        cam2world_44[:3] = camera.camera_to_worlds[0].cpu().numpy()
                        K_44[:3, :3] = camera.get_intrinsics_matrices()[0].cpu().numpy()

                        if self.rescale_to_world:
                            cam2world_44[:3, 3] *= nerfstudio_to_world_scale
                            cam2world_44 = nerfstudio_to_world_transform @ cam2world_44

                        to_save["cam_to_world_44"] = cam2world_44
                        to_save["intrinsics_44"] = K_44

                        np.savez(file=npz_path, **to_save)

                        ### End Regsplatfacto code ##
                        # Note that we have also *removed* a lot of code from SplatFacto's
                        # implementation, particularly around saving alternative outputs e.g. RGBs.

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.split.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(
            Panel(
                table,
                title="[bold][green]:tada: Render on split {} Complete :tada:[/bold]",
                expand=False,
            )
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(DatasetRenderMeshing).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(DatasetRenderMeshing)  # noqa
