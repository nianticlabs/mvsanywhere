import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from mvsanywhere.run_demo import init_model
from mvsanywhere.run_demo import prepare_scan_files, init_model
from mvsanywhere.options import OptionsHandler
from mvsanywhere.utils.dataset_utils import get_dataset
from mvsanywhere.utils.generic_utils import to_gpu
from mvsanywhere.utils.geometry_utils import NormalGenerator


class MVSAnywhereDataset(InputDataset):
    """
    A nerfstudio Dataset which additionally returns depth and normal estimates from MVSAnywhere.

    This will attempt to load the estimates from disk, but if no estimates can be found, it will
    predict them and then save to disk for future iterations.
    """

    depth_folder = "mvsanywhere_predictions"

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        dataset_path: str,
        config_file: str,
        data_config_file: str,
        load_weights_from_checkpoint: str,
        batch_size: int,
        image_width: int,
        image_height: int,
        rotate_images: bool = False,
        scale_factor: float = 1.0
    ) -> None:
        super().__init__(dataparser_outputs=dataparser_outputs, scale_factor=scale_factor)

        self.dataset_path = dataset_path
        self.config_file = config_file
        self.data_config_file = data_config_file
        self.load_weights_from_checkpoint = load_weights_from_checkpoint
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.rotate_images = rotate_images
        self.depth_scale = dataparser_outputs.dataparser_scale
        self.depth_and_normal_predictor = None

    def _predict_all_depths_and_normals(self, depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts a depth and normal map for a single image in the dataset.

        Args:
            image_idx (int): The index of the image to predict for.

        Returns:
            tuple[np.ndarray, np.ndarray]: The predicted depth and normal as numpy arrays.
        """

        option_handler = OptionsHandler()
        option_handler.parse_and_merge_options(self.config_file, True)
        option_handler.options.datasets = [
            option_handler.load_options_from_yaml(self.data_config_file)
        ]
        if self.load_weights_from_checkpoint is not None:
            option_handler.options.load_weights_from_checkpoint = self.load_weights_from_checkpoint
        if self.batch_size is not None:
            option_handler.options.batch_size = self.batch_size
        if self.image_width is not None:
            option_handler.options.image_width = self.image_width
        if self.image_height is not None:
            option_handler.options.image_height = self.image_height

        dataset_path = Path(self.dataset_path)
        option_handler.options.scan_parent_directory = dataset_path.parent
        option_handler.options.scan_name = dataset_path.name
        opts = option_handler.options

        # Init model
        model = init_model(opts)

        # Compute tuples
        dataset_opts = prepare_scan_files(opts)
        opts.datasets = [dataset_opts]

        # Build dataset
        dataset_class, scans = get_dataset(
            dataset_opts.dataset, dataset_opts.dataset_scan_split_file, opts.single_debug_scan_id
        )
        assert len(scans) == 1, "Regsplatfacto only allows one scan at a time."
        scan = scans[0]
        dataset = dataset_class(
            dataset_opts.dataset_path,
            split=dataset_opts.split,
            mv_tuple_file_suffix=dataset_opts.mv_tuple_file_suffix,
            limit_to_scan_id=scan,
            include_full_res_depth=True,
            tuple_info_file_location=dataset_opts.tuple_info_file_location,
            num_images_in_tuple=None,
            shuffle_tuple=opts.shuffle_tuple,
            include_high_res_color=False,
            include_full_depth_K=True,
            skip_frames=opts.skip_frames,
            skip_to_frame=opts.skip_to_frame,
            image_width=opts.image_width,
            image_height=opts.image_height,
            pass_frame_id=True,
            disable_flip=True,
            rotate_images=self.rotate_images,
            matching_scale=opts.matching_scale,
            prediction_scale=opts.prediction_scale,
            prediction_num_scales=opts.prediction_num_scales,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opts.batch_size,
            shuffle=False,
            num_workers=opts.num_workers,
            drop_last=False,
        )

        with torch.inference_mode():

            for batch_ind, batch in enumerate(tqdm(dataloader)):

                cur_data, src_data = batch
                cur_data = to_gpu(cur_data, key_ignores=["frame_id_string"])
                src_data = to_gpu(src_data, key_ignores=["frame_id_string"])

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

                torch.cuda.synchronize()

                Path(depth_path.parent).mkdir(parents=True, exist_ok=True)

                normal_generator = NormalGenerator(
                    height=outputs["depth_pred_s0_b1hw"].shape[2],
                    width=outputs["depth_pred_s0_b1hw"].shape[3],
                    smoothing_kernel_size=0,
                    smoothing_kernel_std=0.,
                )
                normals_pred_b3hw = normal_generator(
                    outputs["depth_pred_s0_b1hw"].cpu().float(),
                    cur_data["invK_s0_b44"].cpu().float(),
                )

                for elem_ind in range(outputs["depth_pred_s0_b1hw"].shape[0]):
                    frame_id = cur_data["frame_id_string"][elem_ind]
                    frame_id = Path(frame_id).stem

                    depth_1hw = outputs["depth_masked_pred_s0_b1hw"][elem_ind].float().cpu().numpy()
                    normal_3hw = normals_pred_b3hw[elem_ind].float().cpu().numpy()

                    if self.rotate_images:
                        # save back as landscape for splatting
                        depth_1hw = np.rot90(depth_1hw, axes=(1, 2))
                        normal_3hw = np.rot90(normal_3hw, axes=(1, 2))
                        rot_z = Rotation.from_euler("z", -90, degrees=True).as_matrix()
                        normal_3hw = np.einsum("ij,jkl->ikl", rot_z, normal_3hw)

                    np.savez(
                        depth_path.parent / f"{frame_id}.npz",
                        depth=depth_1hw,
                        normal=normal_3hw,
                    )

    def _get_depth_and_normal_estimate(self, image_idx: int) -> dict[str, torch.Tensor]:
        """
        Returns the depth and normal estimates for a single image in the dataset. This will
        attempt to load from disk, but if no estimates are found, it will predict them and save
        to disk.

        Args:
            image_idx (int): The index of the image to get the estimates for.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the depth and normal estimates, ready
                to be inserted into the dictionary returned from `super().get_data(...)`.
        """
        image_path = Path(self._dataparser_outputs.image_filenames[image_idx])
        depth_path = image_path.parent.parent / self.depth_folder / f"{image_path.stem}.npz"

        if depth_path.exists():
            data = np.load(depth_path)
            depth_11hw = data["depth"]
            normal_13hw = data["normal"]

        else:
            warnings.warn(
                f"Depth and normal estimates were not found in {depth_path.parent}.\n"
                f"We will now run inference and save to disk at {depth_path.parent}."
            )
            self._predict_all_depths_and_normals(depth_path)
            if depth_path.exists():
                data = np.load(depth_path)
                depth_11hw = data["depth"]
                normal_13hw = data["normal"]
            else:
                raise FileNotFoundError(f"Depth and normal estimates were not saved to {depth_path}.")

        # convert normal to 0-1 and nerfstudio convention
        normal_13hw = (normal_13hw + 1) / 2

        # scale depth from metric to splat scale
        depth_11hw *= self.depth_scale

        return {"depth": torch.tensor(depth_11hw), "normal": torch.tensor(normal_13hw)}

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        data = super().get_data(image_idx=image_idx, image_type=image_type)
        data.update(self._get_depth_and_normal_estimate(image_idx))
        return data
