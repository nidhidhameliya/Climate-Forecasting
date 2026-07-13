import torch
import torch.nn as nn
import json


class ExtremeWeightedMSE(nn.Module):
    def __init__(
        self,
        threshold_path="data/processed/extreme_thresholds.json",
        stats_path="data/processed/mean_std.json",
        extreme_weight=5.0,
    ):
        super(ExtremeWeightedMSE, self).__init__()

        # Default values (in case files don't exist)
        self.threshold = 0.0
        self.extreme_weight = extreme_weight

        # Load thresholds if available
        try:
            with open(threshold_path, "r") as f:
                thresholds = json.load(f)

            with open(stats_path, "r") as f:
                stats = json.load(f)

            threshold_celsius = thresholds["top_5_percent_celsius"]
            mean = stats["mean"]
            std = stats["std"]

            # Convert threshold to normalized space
            self.threshold = (threshold_celsius - mean) / std

        except Exception as e:
            print("⚠️ Warning: Could not load threshold/stat files.")
            print("Using default threshold = 0.")
            print("Error:", e)

    def forward(self, preds, targets):

        extreme_mask = (targets > self.threshold).float()
        weights = 1.0 + self.extreme_weight * extreme_mask

        loss = weights * (preds - targets) ** 2

        return torch.mean(loss)


# ✅ This is what train.py expects
def get_loss(config=None):
    """
    Returns the loss function.
    Uses ExtremeWeightedMSE by default.
    """

    extreme_weight = 5.0

    if config is not None and "extreme_weight" in config:
        extreme_weight = config["extreme_weight"]

    return ExtremeWeightedMSE(extreme_weight=extreme_weight)