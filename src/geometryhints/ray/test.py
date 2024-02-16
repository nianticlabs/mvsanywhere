import os
import sys
import uuid
from argparse import Namespace

from loguru import logger
from ray.job_submission import JobSubmissionClient

import geometryhints.options as options

# When testing & developing with opened K8s port-forward, use "http://localhost:8265":
#  $ kubectl port-forward services/ray-cluster-head-svc 8265:8265
RAY_INSTANCE = "http://geometryhints.ray.masala.eng.nianticlabs.com"


class RayRunner:
    """A class to submit a training job to the Ray Cluster."""

    def __init__(self) -> None:
        """Initializes RayRunner."""
        logger.info("Setting up a connection to the Ray Cluster...")
        self.client = JobSubmissionClient(RAY_INSTANCE)

    def submit_job(self, options: options.Options) -> None:
        """Submits a training job to the Ray Cluster.

        :param options: the options to use during training
        """

        command = "bash ./test_remote.sh"

        logger.info(f"Submitting...")
        submission_id = self._generate_submission_id(options)
        self.client.submit_job(
            submission_id=submission_id,
            entrypoint=command,
            entrypoint_num_gpus=1,
            entrypoint_resources={"a100": 1},
            runtime_env={
                "working_dir": ".",
                "py_modules": ["src/geometryhints"],
                "excludes": [".git", "media/*", "data_splits/*", "*.ckpt"],
            },
        )
        logger.info(f"Job submitted successfully! 🎉 Job ID: {submission_id}")
        logger.info("")
        logger.info("You can inspect your training Job on the Ray Dashboard:")
        logger.info(f"  {RAY_INSTANCE}/#/jobs/{submission_id}")
        logger.info("")
        logger.info("Use the following command to access training logs in real time:")
        logger.info(f'  ray job logs "{submission_id}" --address {RAY_INSTANCE} --follow')
        logger.info("")
        logger.info("Use the following command to stop your Job:")
        logger.info(f'  ray job stop "{submission_id}" --address {RAY_INSTANCE}')
        logger.info("")
        logger.info(
            "To resume your training after a failure, simply submit your training job "
            "again with the same command line arguments."
        )
        logger.info("")

    @staticmethod
    def _generate_submission_id(options: Namespace) -> str:
        """Generates a unique submission ID.

        :param options: the options to use during training
        :return: a unique submission ID
        """
        username = os.getlogin()
        short_hash = str(uuid.uuid4())[:6]
        return f"{username}-{options.name}-{short_hash}"


def main() -> None:
    # Ray training script reuses the same training arguments as the PyTorch Lightning training
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    runner = RayRunner()
    runner.submit_job(options=opts)


if __name__ == "__main__":
    main()
