from pathlib import Path

import lightning as pl
import ray.train.lightning
from lightning.pytorch.loggers import TensorBoardLogger
from ray.train.torch import TorchTrainer

import doubletake.options as options
from doubletake.train import (
    prepare_callbacks,
    prepare_dataloaders,
    prepare_model,
    prepare_trainer,
)

RAY_OUTPUT_PATH = "/mnt/nas3/shared/projects/fmvs/fmvs/logs/ray/"

def _get_ray_trainer(opts: options.Options, is_resume: bool = False) -> pl.Trainer:
    """Get the Ray trainer for training.

    :param options: options to use during training.
    :param data_module: data module to use during training.
    :param is_multidepth: whether the model is a multi-depth model.
    :param disable_sanity_check: whether to disable the sanity check.
    :return: the Ray trainer for training.
    """
    from loguru import logger  # Lazy loading logger to avoid global logger state in Ray

    # prepare callbacks: NOTE that we disable versioning!
    callbacks = prepare_callbacks(opts=opts, enable_version_counter=False, is_resume=is_resume)
    callbacks.append(ray.train.lightning.RayTrainReportCallback())

    ddp_strategy = ray.train.lightning.RayDDPStrategy(
        find_unused_parameters=opts.matching_encoder_type == "unet_encoder"
    )

    # NOTE: check if we have to resume a checkpoint from the log dir.
    # For instance, the training might have be killed (spot instances..),
    # and Kubernets will try to resume it automatically using THE SAME command used at the
    # beginning of the training. It is important that the latest checkpoint is available at a
    # fixed position, so that we can resume the weights, the optimiser etc.

    # set up a tensorboard logger through lightning
    train_logger = TensorBoardLogger(
        save_dir=str((Path(RAY_OUTPUT_PATH)).resolve()),
        name=opts.name
    )

    lightning_trainer = prepare_trainer(
        opts=opts,
        logger=train_logger,
        callbacks=callbacks,
        ddp_strategy=ddp_strategy,
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        auto_devices=True,
    )
    ray_trainer = ray.train.lightning.prepare_trainer(lightning_trainer)
    return ray_trainer


def ray_train_func(opts: options.Options) -> None:
    """Train function to be called by Ray.

    :param options_dict: options to use during training.
    """
    from loguru import logger  # Lazy loading logger to avoid global logger state in Ray

    with logger.catch(reraise=True):
        # prepare the model.
        model = prepare_model(opts=opts)

        # prepare the training data
        train_dataloader, val_dataloaders = prepare_dataloaders(opts=opts)

        # check if we are resuming a model
        resume_ckpt = None
        is_resume = False
        ckpt_path_last = Path(RAY_OUTPUT_PATH) / opts.name / "last.ckpt"

        if ckpt_path_last.exists():
            # in this case we have to resume but it is the first time
            logger.info(f"Found a checkpoint to resume: {ckpt_path_last}")
            resume_ckpt = ckpt_path_last
            is_resume = True

        # prepare the trainer
        ray_trainer = _get_ray_trainer(opts=opts, is_resume=is_resume)

        logger.info(f"Resuming checkpoint: {resume_ckpt}")
        ray_trainer.fit(model, train_dataloader, val_dataloaders, ckpt_path=resume_ckpt)


def main() -> None:
    """Main entrypoint for training with Ray.

    :param options: options to use during training. If None, will parse from command line.
    """
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    scaling_config = ray.train.ScalingConfig(
        num_workers=opts.gpus,
        use_gpu=True,
        resources_per_worker={"a100": 1},
    )

    trainer = TorchTrainer(
        train_loop_per_worker=ray_train_func,
        train_loop_config=opts,
        scaling_config=scaling_config,
        run_config=ray.train.RunConfig(
            # name="ray",
            storage_path=str((Path(RAY_OUTPUT_PATH) / opts.name).resolve()),
            failure_config=ray.train.FailureConfig(max_failures=10),
        ),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
