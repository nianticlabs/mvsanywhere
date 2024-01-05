import os
from pathlib import Path

import geometryhints.options as options
from geometryhints.utils.metrics_utils import ResultsAverager


def print_results(results_path: Path):
    if not results_path.exists():
        raise ValueError(f"Scores output directory {results_path} does not exist.")

    print(f"Scores for {results_path}")

    loaded_metrics = ResultsAverager("", f"frame metrics")
    loaded_metrics = ResultsAverager("", f"scene metrics")

    loaded_metrics.load_scores(results_path)
    loaded_metrics.pretty_print_results(print_running_metrics=False)
    loaded_metrics.print_sheets_friendly(
        include_metrics_names=True,
        print_running_metrics=False,
    )


if __name__ == "__main__":
    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()

    option_handler.parser.add_argument(
        "--results_path",
        type=Path,
        required=True,
        help="Path to the ScanNet dataset.",
    )

    option_handler.parse_and_merge_options(ignore_cl_args=False)

    opts = option_handler.options

    print_results(opts.results_path)
