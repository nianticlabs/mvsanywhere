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
    default=Path("checkpoints/.ckpt"),
)
@click.option(
    "--output-dir",
    help="Path to the output directory where meshes and 2D predictions will be saved",
    type=Path,
    default=Path("results"),
)
def evalute_incremental_render(checkpoint: str, output_dir: Path):
    logger.info("Generating TSDFs and 2D embeddings using our model for each scan")
    subprocess.run(
        [
            "python",
            "-m",
            "doubletake.test_incremental_render",
            f"load_weights_from_checkpoint={str(checkpoint)}",
            f"output_base_path={output_dir}",
            "cache_depths=False",
        ]
    )
