"""
Creates sample dataset given original dataset path
"""
import argparse
import os
import random
import shutil
from pathlib import Path


def create_sample_dataset(source_dataset_path: str, destination_dataset_path: str, sample_size: int) -> None:
    """
    Creates sample dataset from original dataset

    Args:
        source_dataset_path: Source Dataset Path
        destination_dataset_path: Destination Dataset Path
        sample_size: Sample size
    """
    filenames = os.listdir(source_dataset_path)
    sampled_filenames = random.sample(filenames, sample_size)
    sampled_filepaths = [os.path.join(source_dataset_path, filename) for filename in sampled_filenames]
    destination_filepaths = [os.path.join(destination_dataset_path, filename) for filename in sampled_filenames]

    Path(destination_dataset_path).mkdir(exist_ok=True, parents=True)
    for src_filepath, dst_filepath in zip(sampled_filepaths, destination_filepaths):
        shutil.copy(src_filepath, dst_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Sample Dataset')
    parser.add_argument('--src', type=str, help='source dataset path')
    parser.add_argument('--dst', type=str, help='destination dataset path')
    parser.add_argument('--size', type=int, help='sample size')

    args = parser.parse_args()
    create_sample_dataset(args.src, args.dst, args.size)
