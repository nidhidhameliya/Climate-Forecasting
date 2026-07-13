import torch
import json
from training.metrics import rmse, extreme_rmse, hit_rate


def test(model, loader, threshold_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(threshold_path) as f:
        threshold = json.load(f)["top_5_percent_celsius"]

    model.eval()

    overall = 0
    extreme = 0
    hits = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            # Fix tensor shape for ConvLSTM: (batch, lat, time, channel, lon) -> (batch, time, channel, height, width)
            X = X.permute(0, 2, 3, 1, 4)

            preds = model(X.float())

            overall += rmse(preds, y.float()).item()
            extreme += extreme_rmse(preds, y.float(), threshold).item()
            hits += hit_rate(preds, y.float(), threshold).item()

    return {
        "overall_rmse": overall / len(loader),
        "extreme_rmse": extreme / len(loader),
        "hit_rate": hits / len(loader)
    }