import torch
from training.metrics import rmse


def validate(model, loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_rmse = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            # Fix tensor shape for ConvLSTM: (batch, lat, time, channel, lon) -> (batch, time, channel, height, width)
            X = X.permute(0, 2, 3, 1, 4)

            preds = model(X.float())
            total_rmse += rmse(preds, y.float()).item()

    return total_rmse / len(loader)