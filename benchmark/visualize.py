"""Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/visualize.py"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the benchmark results")
    parser.add_argument(
        "--file",
        type=str,
        default="all_perf.csv",
        help="Path to the CSV file with benchmark resutls (default: all_perf.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = args.file

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        exit(1)

    names = df["name"].unique()

    for name in names:
        subset = df[df["name"] == name]
        plt.plot(subset["seqlen"], subset["bw"], label=name)

    plt.title("bandwidth")
    plt.xlabel("seqlen")
    plt.ylabel("bw (GB/s)")
    plt.legend()
    plt.savefig(f"{file_path.split('.')[0].split('/')[-1]}_bandwidth_vs_seqlen.svg", format="svg")
