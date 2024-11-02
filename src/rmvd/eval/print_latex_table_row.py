from rmvd.eval.robust_mvd_benchmark import RobustMultiViewDepthBenchmark
import pandas as pd
from pathlib import Path
import argparse


def print_latex_table_row_for_path(csv_results_path: Path):
    # Load all the csv files into a single pandas dataframe
    combined_results = pd.read_csv(csv_results_path)

    # Create a MultiIndex from 'dataset' and 'metric' columns
    combined_results.set_index(['dataset', 'metric'], inplace=True)

    # Sort the index to make it cleaner
    combined_results.sort_index(inplace=True)
    combined_results = combined_results['0']

    RobustMultiViewDepthBenchmark.print_latex_table(combined_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prints a latex table row corresponding to a results dir"
    )
    parser.add_argument('csv_results_path', type=str, help='The combined results.csv file')
    csv_results_path = Path(parser.parse_args().csv_results_path)

    print_latex_table_row_for_path(csv_results_path=csv_results_path)