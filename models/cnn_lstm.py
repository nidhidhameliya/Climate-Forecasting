import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config["model"]["hidden_dim"]

        self.cnn = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        spatial_feats = []

        for t in range(T):
            feat = self.cnn(x[:, t])
            feat = feat.mean(dim=[2,3])  # global pooling
            spatial_feats.append(feat)

        seq = torch.stack(spatial_feats, dim=1)

        lstm_out, _ = self.lstm(seq)

        last = lstm_out[:, -1]

        out = self.output(last)

        return out.view(B, 1, 1, 1)