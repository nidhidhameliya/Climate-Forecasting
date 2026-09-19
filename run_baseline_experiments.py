"""Train and compare the requested next-day forecasting baselines.

All learned models use the same saved train/validation/test tensors, sequence
length, forecast horizon, optimizer, loss, and evaluation protocol. Persistence
uses the final input day as the prediction and requires no training.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from models.model_utils import get_model
from training.losses import get_loss


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "experiments" / "reviewer_baselines"
MODEL_NAMES = ("persistence", "lstm", "cnn_lstm", "convlstm", "transformer")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config() -> dict:
    with (ROOT / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


def load_data() -> dict[str, np.ndarray]:
    base = ROOT / "data" / "processed" / "tensors"
    return {
        f"{split}_{kind}": np.load(base / f"{split}_{kind}.npy")
        for split in ("train", "val", "test")
        for kind in ("X", "y")
    }


def calculate_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    return {
        "rmse": float(np.sqrt(mean_squared_error(target_flat, prediction_flat))),
        "mae": float(mean_absolute_error(target_flat, prediction_flat)),
        "r2": float(r2_score(target_flat, prediction_flat)),
        "correlation": float(np.corrcoef(prediction_flat, target_flat)[0, 1]),
    }


def denormalize(array: np.ndarray, mean: float, std: float) -> np.ndarray:
    return array * std + mean


def predict_model(model, loader, device, mean: float, std: float) -> tuple[np.ndarray, np.ndarray]:
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for inputs, batch_targets in loader:
            output = model(inputs.to(device)).cpu().numpy()[:, 0]
            predictions.append(output)
            targets.append(batch_targets.numpy()[:, 0])
    return denormalize(np.concatenate(predictions), mean, std), denormalize(np.concatenate(targets), mean, std)


def run_persistence(data: dict[str, np.ndarray], mean: float, std: float) -> dict:
    prediction = denormalize(data["test_X"][:, -1, 0], mean, std)
    target = denormalize(data["test_y"][:, 0], mean, std)
    return {
        "model": "Persistence",
        "class": "last-input-day",
        "trainable_parameters": 0,
        "epochs": 0,
        "validation": calculate_metrics(
            denormalize(data["val_X"][:, -1, 0], mean, std),
            denormalize(data["val_y"][:, 0], mean, std),
        ),
        "test": calculate_metrics(prediction, target),
    }


def run_learned_model(name: str, config: dict, data: dict[str, np.ndarray], mean: float, std: float, device, epochs: int, batch_size: int, output_dir: Path) -> dict:
    model_config = {**config, "model": {**config["model"], "name": name}}
    model = get_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    loss_fn = get_loss(config["training"])
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(data["train_X"]).float(), torch.from_numpy(data["train_y"]).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(data["val_X"]).float(), torch.from_numpy(data["val_y"]).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(data["test_X"]).float(), torch.from_numpy(data["test_y"]).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            output = model(inputs.to(device))
            loss = loss_fn(output, targets.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                validation_loss += float(loss_fn(model(inputs.to(device)), targets.to(device)).item())
        history.append({"epoch": epoch + 1, "train_loss": running_loss / len(train_loader), "validation_loss": validation_loss / len(val_loader)})

    validation_prediction, validation_target = predict_model(model, val_loader, device, mean, std)
    test_prediction, test_target = predict_model(model, test_loader, device, mean, std)
    checkpoint = output_dir / f"{name}.pth"
    torch.save(model.state_dict(), checkpoint)
    return {
        "model": name,
        "class": model.__class__.__name__,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "learning_rate": config["training"]["learning_rate"],
        "loss": "ExtremeWeightedMSE",
        "training_seconds": round(time.perf_counter() - start, 3),
        "history": history,
        "validation": calculate_metrics(validation_prediction, validation_target),
        "test": calculate_metrics(test_prediction, test_target),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config()
    data = load_data()
    stats = json.loads((ROOT / "data" / "processed" / "mean_std.json").read_text())
    mean, std = float(stats["mean"]), float(stats["std"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    requested = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    if "persistence" in requested:
        results.append(run_persistence(data, mean, std))
    for name in requested:
        if name == "persistence":
            continue
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{name}'. Choose from {', '.join(MODEL_NAMES)}.")
        results.append(run_learned_model(name, config, data, mean, std, device, args.epochs, args.batch_size, output_dir))

    result_path = output_dir / "comparison_results.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        merged = {result["model"].lower(): result for result in previous.get("results", [])}
        merged.update({result["model"].lower(): result for result in results})
        results = [merged[name] for name in MODEL_NAMES if name in merged]

    report = {
        "protocol": {
            "sequence_length": config["sequence_length"],
            "forecast_horizon_days": 1,
            "train_samples": int(data["train_X"].shape[0]),
            "validation_samples": int(data["val_X"].shape[0]),
            "test_samples": int(data["test_X"].shape[0]),
            "spatial_shape": list(data["test_y"].shape[-2:]),
            "device": str(device),
            "seed": args.seed,
            "metric_space": "same denormalized stored temperature scale for every model",
        },
        "results": results,
    }
    result_path.write_text(json.dumps(report, indent=2))
    rows = []
    for result in results:
        rows.append({"model": result["model"], "trainable_parameters": result["trainable_parameters"], "test_rmse": result["test"]["rmse"], "test_mae": result["test"]["mae"], "test_r2": result["test"]["r2"], "test_correlation": result["test"]["correlation"]})
    import csv
    with (output_dir / "comparison_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()