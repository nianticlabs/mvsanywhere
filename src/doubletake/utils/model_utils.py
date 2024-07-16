import torch
from loguru import logger

import doubletake.modules.feature_volume as feature_volume
import doubletake.modules.mesh_hint_volume as mesh_hint_volume
from doubletake.experiment_modules.doubletake_model import DepthModelCVHint
from doubletake.experiment_modules.sr_depth_model import DepthModel


def get_model_class(opts):
    if opts.model_type == "depth_model":
        model_class_to_use = DepthModel
    elif opts.model_type == "cv_hint_depth_model":
        model_class_to_use = DepthModelCVHint
    else:
        raise ValueError(f"Unknown model type: {opts.model_type}")
    return model_class_to_use


def load_model_inference(opts, model_class_to_use):
    try:
        model = model_class_to_use.load_from_checkpoint(
            opts.load_weights_from_checkpoint, args=None
        )
    except:
        logger.info("Failed to load model normally. Using manual loading via state_dict.")
        model = model_class_to_use(opts)
        model.load_state_dict(torch.load(opts.load_weights_from_checkpoint)["state_dict"])

    if opts.fast_cost_volume and (
        isinstance(model.cost_volume, feature_volume.FeatureVolumeManager)
        or isinstance(model.cost_volume, mesh_hint_volume.FeatureMeshHintVolumeManager)
    ):
        model.cost_volume = model.cost_volume.to_fast()
    return model


def load_model_training(opts, model_class_to_use):
    """Loads a model for training."""
    if opts.load_weights_from_checkpoint is not None:
        logger.info(f"Loading weights from {opts.load_weights_from_checkpoint}.")
        model = model_class_to_use.load_from_checkpoint(
            opts.load_weights_from_checkpoint,
            opts=opts,
            args=None,
        )
    elif opts.lazy_load_weights_from_checkpoint is not None:
        logger.info(f"Lazy loading weights from {opts.lazy_load_weights_from_checkpoint}.")
        model = model_class_to_use(opts)
        state_dict = torch.load(opts.lazy_load_weights_from_checkpoint)["state_dict"]
        available_keys = list(state_dict.keys())
        for param_key, param in model.named_parameters():
            if param_key in available_keys:
                try:
                    if isinstance(state_dict[param_key], torch.nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = state_dict[param_key].data
                    else:
                        param = state_dict[param_key]

                    model.state_dict()[param_key].copy_(param)
                except:
                    logger.info(f"WARNING: could not load weights for {param_key}")
    else:
        logger.info("No weights loaded. Instantiating new model.")
        model = model_class_to_use(opts)

    return model
