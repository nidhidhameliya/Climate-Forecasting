"""
Creates sliding window sequences from the preprocessed, normalized data splits.

This script loads the train, validation, and test NumPy arrays, transforms them
into sequences of a specified length, and saves them as model-ready tensors.
The input `X` will have a shape of (num_samples, sequence_length, channels, height, width),
and the target `y` will have a shape of (num_samples, channels, height, width).

The `sequence_length` is read from the main `config.yaml` file.
"""

import numpy as np
import yaml
import os
from typing import Tuple


def create_sequences(data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates input (X) and target (y) sequences from a time-series dataset.

    Args:
        data: A NumPy array of shape (time, height, width).
        seq_len: The number of time steps in each input sequence.

    Returns:
        A tuple containing:
        - X: Input sequences, shape (num_samples, seq_len, height, width).
        - y: Target values, shape (num_samples, height, width).
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def main() -> None:
    """
    Main function to load data, create sequences, and save them to disk.
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    seq_len = config["sequence_length"]

    output_dir = "data/processed/tensors"
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"Creating sequences for {split}...")
        data_path = f"data/processed/{split}.npy"
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found at {data_path}. Skipping.")
            continue

        data = np.load(data_path)
        X, y = create_sequences(data, seq_len)

        # Reshape for ConvLSTM input: add a channel dimension.
        # (samples, time, H, W) -> (samples, time, C, H, W)
        X = X[:, :, np.newaxis, :, :]
        # (samples, H, W) -> (samples, C, H, W)
        y = y[:, np.newaxis, :, :]

        np.save(os.path.join(output_dir, f"{split}_X.npy"), X)
        np.save(os.path.join(output_dir, f"{split}_y.npy"), y)

    print("✅ Sequences created successfully.")


if __name__ == "__main__":
    main()