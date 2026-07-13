import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_numpy_tensors(split):

    base_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "tensors"
    )

    X_path = os.path.join(base_path, f"{split}_X.npy")
    y_path = os.path.join(base_path, f"{split}_y.npy")

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Missing file: {X_path}")

    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing file: {y_path}")

    X = np.load(X_path)  # (B, T, H, W, 1)
    y = np.load(y_path)  # (B, H, W, 1)

    # -------------------------------------------------
    # Fix X shape: (B, T, H, W, C) → (B, T, C, H, W)
    # -------------------------------------------------
    X = np.transpose(X, (0, 1, 4, 2, 3))

    # -------------------------------------------------
    # Fix y shape: (B, H, W, C) → (B, C, H, W)
    # -------------------------------------------------
    y = np.transpose(y, (0, 3, 1, 2))

    # Convert to float32 tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y


def get_dataloaders(config):

    batch_size = config["training"]["batch_size"]

    train_X, train_y = load_numpy_tensors("train")
    val_X, val_y = load_numpy_tensors("val")

    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def get_test_loader(config):

    batch_size = config["training"]["batch_size"]

    test_X, test_y = load_numpy_tensors("test")

    test_dataset = TensorDataset(test_X, test_y)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_loader