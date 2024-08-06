import subprocess
from pathlib import Path

import click
from loguru import logger


@click.group()
def run():
    pass


@run.command()
@click.option(
    "--checkpoint",
    help="Path to the model checkpoint to use",
    type=Path,
    default=Path("checkpoints/doubletake_model.ckpt"),
)
@click.option(
    "--output-dir",
    help="Path to the output directory where meshes and 2D predictions will be saved",
    type=Path,
    default=Path("results"),
)
def incremental(checkpoint: str, output_dir: Path):
    logger.info("Evaluating the model in incremental mode")
    subprocess.run(
        [
            "python",
            "-m",
            "doubletake.test_incremental",
            "--config_file",
            "configs/models/doubletake_model.yaml",
            "--data_config",
            "configs/data/scannet/scannet_default_test.yaml",
            "--load_weights_from_checkpoint",
            str(checkpoint),
            "--batch_size",
            str(1),
            "--output_base_path",
            f"{str(output_dir)}",
            "--depth_hint_aug",  ## we want to use hints
            str(0.0),
            "--load_empty_hint",  ## we don't have hints in advance
            "--name",
            "incremental",
            "--run_fusion",  ## we want to build the mesh while we go
            "--fusion_resolution",
            str(0.02),
            "--fusion_max_depth",
            str(3.5),
            "--extended_neg_truncation",
            "--num_workers",
            str(12),
        ],
        check=True,
    )


if __name__ == "__main__":
    run()
