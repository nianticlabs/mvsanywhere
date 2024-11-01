"""
    Trains a DepthModel model. Uses an MVS dataset from datasets.

    - Outputs logs and checkpoints to opts.log_dir/opts.name
    - Supports mixed precision training by setting '--precision 16'

    We train with a batch_size of 16 with 16-bit precision on two A100s.

    Example command to train with two GPUs
        python train.py --name HERO_MODEL \
                    --log_dir logs \
                    --config_file configs/models/hero_model.yaml \
                    --data_config configs/data/scannet_default_train.yaml \
                    --gpus 2 \
                    --batch_size 16;

"""


import os
from pathlib import Path
from typing import List, Optional, Tuple, Type, Callable, Union

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, Strategy
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data._utils.collate import default_collate_fn_map
import doubletake.options as options
from doubletake.utils.dataset_utils import get_dataset
from doubletake.utils.generic_utils import copy_code_state
from doubletake.utils.model_utils import get_model_class
import collections
import contextlib
import copy
import re
from typing import Callable, Dict, Optional, Tuple, Type, Union

import torch



default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)

def collate(
    batch,
    *,
    collate_fn_map: Optional[dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    r"""
    General collate function that handles collection type of element within each batch.

    The function also opens function registry to deal with specific element types. `default_collate_fn_map`
    provides default collate functions for tensors, numpy arrays, numbers and strings.

    Args:
        batch: a single batch to be collated
        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
            If the element type isn't present in this dictionary,
            this function will go through each key of the dictionary in the insertion order to
            invoke the corresponding collate function if the element type is a subclass of the key.

    Examples:
        >>> def collate_tensor_fn(batch, *, collate_fn_map):
        ...     # Extend this function to handle batch of tensors
        ...     return torch.stack(batch, 0)
        >>> def custom_collate(batch):
        ...     collate_map = {torch.Tensor: collate_tensor_fn}
        ...     return collate(batch, collate_fn_map=collate_map)
        >>> # Extend `default_collate` by in-place modifying `default_collate_fn_map`
        >>> default_collate_fn_map.update({torch.Tensor: collate_tensor_fn})

    Note:
        Each collate function requires a positional argument for batch and a keyword argument
        for the dictionary of collate functions as `collate_fn_map`.
    """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](
                    batch, collate_fn_map=collate_fn_map
                )

    if isinstance(elem, collections.abc.Mapping):
        try:
            if isinstance(elem, collections.abc.MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(elem)
                dd = {}
                for key in elem:
                    print(key)
                    dd[key] = collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
                clone.update(dd)

                # clone.update(
                #     {
                #         key: collate(
                #             [d[key] for d in batch], collate_fn_map=collate_fn_map
                #         )
                #         for key in elem
                #     }
                # )
                return clone
            else:
                return elem_type(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {
                key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
                for key in elem
            }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in zip(*batch)
            )
        )
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                if isinstance(elem, collections.abc.MutableSequence):
                    # The sequence type may have extra properties, so we can't just
                    # use `type(data)(...)` to create the new sequence.
                    # Create a clone and update it if the sequence type is mutable.
                    clone = copy.copy(elem)  # type: ignore[arg-type]
                    for i, samples in enumerate(transposed):
                        clone[i] = collate(samples, collate_fn_map=collate_fn_map)
                    return clone
                else:
                    return elem_type(
                        [
                            collate(samples, collate_fn_map=collate_fn_map)
                            for samples in transposed
                        ]
                    )
            except TypeError:
                # The sequence type may not support `copy()` / `__setitem__(index, item)`
                # or `__init__(iterable)` (e.g., `range`).
                return [
                    collate(samples, collate_fn_map=collate_fn_map)
                    for samples in transposed
                ]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def custom_collate(batch):
    return collate(batch, collate_fn_map=default_collate_fn_map)


def prepare_dataloaders(opts: options.Options) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Prepare training and validation dataloaders/
    Training loader is one, while we might have multiple dataloaders for validations.
    For instance, we might validate using a different augmentation for hints (always given, never
    given, given with 50% chances etc).

    Params:
        opts: options for the current run
    Returns:
        a train dataloader, a list of dataloaders for validation
    """
    # load dataset and dataloaders
    train_datasets, val_datasets = [], []
    for dataset in opts.datasets:
        dataset_class, _ = get_dataset(
            dataset.dataset, dataset.dataset_scan_split_file, opts.single_debug_scan_id
        )

        train_dataset = dataset_class(
            dataset.dataset_path,
            split="train",
            mv_tuple_file_suffix=dataset.mv_tuple_file_suffix,
            num_images_in_tuple=opts.num_images_in_tuple,
            tuple_info_file_location=dataset.tuple_info_file_location,
            image_width=opts.image_width,
            image_height=opts.image_height,
            shuffle_tuple=opts.shuffle_tuple,
            matching_scale=opts.matching_scale,
            prediction_scale=opts.prediction_scale,
            prediction_num_scales=opts.prediction_num_scales
        )
        train_datasets.append(train_dataset)


    for dataset in opts.val_datasets:
        dataset_class, _ = get_dataset(
            dataset.dataset, dataset.dataset_scan_split_file, opts.single_debug_scan_id
        )
        val_dataset = dataset_class(
            dataset.dataset_path,
            split="val",
            mv_tuple_file_suffix=dataset.mv_tuple_file_suffix,
            num_images_in_tuple=opts.num_images_in_tuple,
            tuple_info_file_location=dataset.tuple_info_file_location,
            image_width=opts.val_image_width,
            image_height=opts.val_image_height,
            include_full_res_depth=opts.high_res_validation,
            matching_scale=opts.matching_scale,
            prediction_scale=opts.prediction_scale,
            prediction_num_scales=opts.prediction_num_scales
        )
        val_datasets.append(val_dataset)

    train_dataloader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=custom_collate,
    )

    val_dataloader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=opts.val_batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=custom_collate,
    )

    return train_dataloader, val_dataloader


def prepare_callbacks(
    opts: options.Options, enable_version_counter: bool = True, is_resume: bool = False
) -> List[pl.pytorch.callbacks.Callback]:
    """Prepare callbacks for the training.
    In our case, callbacks are the strategy used to save checkpoints during training and the
    learning rate monitoring.

    Params:
        opts: options for the current run
        enable_version_counter: if True, save checkpoints with lightning versioning
    Returns:
        a list of callbacks
    """
    # set a checkpoint callback for lignting to save model checkpoints
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor="val_metrics/a5",
        mode="max",
        dirpath=str((Path(opts.log_dir) / opts.name).resolve()),
    )

    # keep track of changes in learning rate
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]
    return callbacks


def prepare_model(opts: options.Options) -> torch.nn.Module:
    """Prepare model to train.
    The function selects the right model given the model class, and eventually resumes the model
    from a checkpoint if `load_weights_from_checkpoint` or `lazy_load_weights_from_checkpoint`
    are set.

    Params:
        opts: options for the current run
    Returns:
        (resumed) model to train
    """
    model_class_to_use = get_model_class(opts)

    if opts.load_weights_from_checkpoint is not None:
        model = model_class_to_use.load_from_checkpoint(
            opts.load_weights_from_checkpoint,
            opts=opts,
            args=None,
        )
    elif opts.lazy_load_weights_from_checkpoint is not None:
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
                    print('Param copied: ', param_key)
                except:
                    print(f"WARNING: could not load weights for {param_key}")
    else:
        # load model using read options
        model = model_class_to_use(opts)
    return model


def prepare_ddp_strategy(opts: options.Options) -> Strategy:
    """Prepare the strategy for data parallel. It defines how to manage multiple processes
    over one or multiple nodes.

    Params:
        opts: options for the current run
    Returns:
        data parallel strategy
    """
    # allowing the lightning DDPPlugin to ignore unused params.
    find_unused_parameters = (opts.matching_encoder_type == "unet_encoder") or ("dinov2" in opts.image_encoder_name) or ("depth_anything" in opts.depth_decoder_name)
    return DDPStrategy(find_unused_parameters=find_unused_parameters)


def prepare_trainer(
    opts: options.Options,
    logger: pl.pytorch.loggers.logger.Logger,
    callbacks: List[pl.pytorch.callbacks.Callback],
    ddp_strategy: Strategy,
    plugins: List[pl.pytorch.plugins.PLUGIN_INPUT] = None,
    resume_ckpt: Optional[str] = None,
    auto_devices: bool = False,
) -> pl.pytorch.trainer.trainer.Trainer:
    """
    Prepare a trainer for the run.
    Params:
        opts: options for the current run
        logger: selected pl logger to use for logging
        callbacks: callbacks for the trainer (such as LRMonitor, Checkpoint saving strategy etc)
        ddp_strategy: strategy for data parallel plugins
        plugins: optional plugins in case of clusters. Default is none because we use a single machine
    Returns:
        (resumed) model to train
    """
    devices = "auto" if auto_devices else opts.gpus

    trainer = pl.Trainer(
        devices=devices,
        log_every_n_steps=opts.log_interval,
        val_check_interval=opts.val_interval,
        limit_val_batches=opts.val_batches,
        max_steps=opts.max_steps,
        precision=opts.precision,
        benchmark=True,
        logger=logger,
        sync_batchnorm=False,
        callbacks=callbacks,
        num_sanity_val_steps=opts.num_sanity_val_steps,
        strategy=ddp_strategy,
        plugins=plugins,
        limit_train_batches=10000,
        profiler="simple",
    )
    return trainer


def main(opts):
    # set seed
    pl.seed_everything(opts.random_seed)

    # prepare model
    model = prepare_model(opts=opts)

    # prepare dataloaders
    train_dataloader, val_dataloaders = prepare_dataloaders(opts=opts)

    # set up a tensorboard logger through lightning
    logger = TensorBoardLogger(save_dir=opts.log_dir, name=opts.name)

    # This will copy a snapshot of the code (minus whatever is in .gitignore)
    # into a folder inside the main log directory.
    copy_code_state(path=os.path.join(logger.log_dir, "code"))

    # dumping a copy of the config to the directory for easy(ier)
    # reproducibility.
    options.OptionsHandler.save_options_as_yaml(
        os.path.join(logger.log_dir, "config.yaml"),
        opts,
    )

    # prepare ddp strategy
    ddp_strategy = prepare_ddp_strategy(opts=opts)

    # prepare callbacks
    callbacks = prepare_callbacks(opts=opts)

    # prepare trainer
    trainer = prepare_trainer(
        opts=opts,
        logger=logger,
        callbacks=callbacks,
        ddp_strategy=ddp_strategy,
    )

    # start training
    trainer.fit(model, train_dataloader, val_dataloaders, ckpt_path=opts.resume)


if __name__ == "__main__":
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
