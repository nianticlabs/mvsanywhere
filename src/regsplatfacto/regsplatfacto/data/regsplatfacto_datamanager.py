from dataclasses import dataclass, field
from typing import Generic, Type

from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)

from regsplatfacto.regsplatfacto.data.mvsanywhere_dataset import MVSAnywhereDataset


@dataclass
class RegSplatfactoDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: RegSplatfactoDatamanager)
    depth_model: str = "mvsanywhere"
    config_file: str = "configs/models/mvsanywhere_dot_model.yaml"
    data_config_file: str = "configs/data/nerfstudio/nerfstudio_empty.yaml"
    load_weights_from_checkpoint: str = None
    batch_size: int = None
    image_width: int = None
    image_height: int = None
    rotate_images: bool = False


class RegSplatfactoDatamanager(FullImageDatamanager):
    """
    Data manager for RegSplatfacto, whis is used to force the dataset to be Metric3dDataset.

    This data manager should be used in the config class when running RegSplatfacto.
    """

    config: RegSplatfactoDatamanagerConfig

    def __init__(self, config: RegSplatfactoDatamanagerConfig, **kwargs):
        self._dataset_type = self._depth_model_to_dataset(config.depth_model)
        super().__init__(config, **kwargs)

    def _depth_model_to_dataset(self, depth_model: str):
        if depth_model == "mvsanywhere":
            return MVSAnywhereDataset
        else:
            raise ValueError(f"Unknown depth model: {depth_model}.")

    @property
    def dataset_type(self):
        return self._dataset_type
    
    def create_train_dataset(self):
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            dataset_path=self.config.data,
            config_file=self.config.config_file,
            data_config_file=self.config.data_config_file,
            load_weights_from_checkpoint=self.config.load_weights_from_checkpoint,
            batch_size=self.config.batch_size,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
            rotate_images=self.config.rotate_images,
        )

    def create_eval_dataset(self):
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            dataset_path=self.config.data,
            config_file=self.config.config_file,
            data_config_file=self.config.data_config_file,
            load_weights_from_checkpoint=self.config.load_weights_from_checkpoint,
            batch_size=self.config.batch_size,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )