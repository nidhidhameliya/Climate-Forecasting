import os
import torch

from training.validate import validate
from training.losses import get_loss


def train_model(model, train_loader, val_loader, optimizer, training_cfg):
    """Basic training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = get_loss(training_cfg)
    epochs = training_cfg.get("epochs", 1)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            # Fix tensor shape for ConvLSTM: (batch, lat, time, channel, lon) -> (batch, channel, time, height, width)
            X = X.permute(0, 2, 3, 1, 4)


            optimizer.zero_grad()
            preds = model(X.float())
            loss = loss_fn(preds, y.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = validate(model, val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

    os.makedirs("experiments/latest", exist_ok=True)
    torch.save(model.state_dict(), "experiments/latest/model.pth")


if __name__ == "__main__":
    from models.model_utils import get_model
    # use correct module name matching file `data_loader/data_loader.py`
    from data_loader.data_loader import get_dataloaders
    import torch
    import yaml
    import os

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # pass entire config dict; dataloader extracts batch_size internally
    train_loader, val_loader = get_dataloaders(config)

    model = get_model(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    train_model(model, train_loader, val_loader, optimizer, config["training"])