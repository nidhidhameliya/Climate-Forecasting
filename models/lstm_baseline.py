"""Per-grid-cell LSTM baseline for next-day temperature maps."""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """Apply one shared temporal LSTM independently to every grid cell."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = config["model"].get("lstm_hidden_dim", config["model"]["hidden_dim"])
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch, timesteps, channels, height, width = x.shape
        sequence = x[:, :, 0].permute(0, 2, 3, 1).reshape(batch * height * width, timesteps, 1)
        encoded, _ = self.lstm(sequence)
        prediction = self.output(encoded[:, -1]).reshape(batch, height, width)
        return prediction.unsqueeze(1)