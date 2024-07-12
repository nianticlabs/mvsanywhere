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
from typing import List, Optional, Tuple

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, Strategy
from torch.utils.data import DataLoader

import doubletake.options as options
from doubletake.utils.dataset_utils import get_dataset
from doubletake.utils.generic_utils import copy_code_state
from doubletake.utils.model_utils import get_model_class


def prepare_dataloaders(opts: options.Options) -> Tuple[DataLoader, List[DataLoader]]:
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
    dataset_class, _ = get_dataset(
        opts.dataset, opts.dataset_scan_split_file, opts.single_debug_scan_id
    )

    train_dataset = dataset_class(
        opts.dataset_path,
        split="train",
        mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
        num_images_in_tuple=opts.num_images_in_tuple,
        tuple_info_file_location=opts.tuple_info_file_location,
        image_width=opts.image_width,
        image_height=opts.image_height,
        shuffle_tuple=opts.shuffle_tuple,
        fill_depth_hints=opts.fill_depth_hints,
        depth_hint_aug=opts.depth_hint_aug,
        depth_hint_dir=opts.depth_hint_dir,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_dataloaders = []
    if opts.fill_depth_hints:
        val_dataset = dataset_class(
            opts.dataset_path,
            split="val",
            mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
            num_images_in_tuple=opts.num_images_in_tuple,
            tuple_info_file_location=opts.tuple_info_file_location,
            image_width=opts.image_width,
            image_height=opts.image_height,
            include_full_res_depth=opts.high_res_validation,
            fill_depth_hints=opts.fill_depth_hints,
            depth_hint_aug=0.5,
            depth_hint_dir=opts.depth_hint_dir,
        )

        val_dataloaders.append(
            DataLoader(
                val_dataset,
                batch_size=opts.val_batch_size,
                shuffle=False,
                num_workers=max(opts.num_workers // 2, 1),
                pin_memory=True,
                drop_last=True,
                persistent_workers=False,
            )
        )

        val_dataset = dataset_class(
            opts.dataset_path,
            split="val",
            mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
            num_images_in_tuple=opts.num_images_in_tuple,
            tuple_info_file_location=opts.tuple_info_file_location,
            image_width=opts.image_width,
            image_height=opts.image_height,
            include_full_res_depth=opts.high_res_validation,
            fill_depth_hints=opts.fill_depth_hints,
            depth_hint_aug=1.0,
            depth_hint_dir=opts.depth_hint_dir,
        )

        val_dataloaders.append(
            DataLoader(
                val_dataset,
                batch_size=opts.val_batch_size,
                shuffle=False,
                num_workers=max(opts.num_workers // 2, 1),
                pin_memory=True,
                drop_last=True,
                persistent_workers=False,
            )
        )

        val_dataset = dataset_class(
            opts.dataset_path,
            split="val",
            mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
            num_images_in_tuple=opts.num_images_in_tuple,
            tuple_info_file_location=opts.tuple_info_file_location,
            image_width=opts.image_width,
            image_height=opts.image_height,
            include_full_res_depth=opts.high_res_validation,
            fill_depth_hints=opts.fill_depth_hints,
            depth_hint_aug=0.0,
            depth_hint_dir=opts.depth_hint_dir,
        )

        val_dataloaders.append(
            DataLoader(
                val_dataset,
                batch_size=opts.val_batch_size,
                shuffle=False,
                num_workers=max(opts.num_workers // 2, 1),
                pin_memory=True,
                drop_last=True,
                persistent_workers=False,
            )
        )

        val_dataset = dataset_class(
            opts.dataset_path,
            split="val",
            mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
            num_images_in_tuple=opts.num_images_in_tuple,
            tuple_info_file_location=opts.tuple_info_file_location,
            image_width=opts.image_width,
            image_height=opts.image_height,
            include_full_res_depth=opts.high_res_validation,
            fill_depth_hints=opts.fill_depth_hints,
            depth_hint_aug=0.0,
            depth_hint_dir=opts.depth_hint_dir,
        )

        val_dataloaders.append(
            DataLoader(
                val_dataset,
                batch_size=opts.val_batch_size,
                shuffle=False,
                num_workers=max(opts.num_workers // 2, 1),
                pin_memory=True,
                drop_last=True,
                persistent_workers=False,
            )
        )
    else:
        val_dataset = dataset_class(
            opts.dataset_path,
            split="val",
            mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
            num_images_in_tuple=opts.num_images_in_tuple,
            tuple_info_file_location=opts.tuple_info_file_location,
            image_width=opts.image_width,
            image_height=opts.image_height,
            include_full_res_depth=opts.high_res_validation,
        )

        val_dataloaders.append(
            DataLoader(
                val_dataset,
                batch_size=opts.val_batch_size,
                shuffle=False,
                num_workers=opts.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )
        )
    return train_dataloader, val_dataloaders


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
        monitor="val_0_metrics/a5" if opts.fill_depth_hints else "val_metrics/a5",
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
    find_unused_parameters = opts.matching_encoder_type == "unet_encoder"
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
