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

        self.decoder = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        spatial_feats = []
        spatial_maps = []

        for t in range(T):
            feat = self.cnn(x[:, t])
            spatial_maps.append(feat)
            spatial_feats.append(feat.mean(dim=[2, 3]))

        seq = torch.stack(spatial_feats, dim=1)

        lstm_out, _ = self.lstm(seq)

        temporal_context = lstm_out[:, -1].unsqueeze(-1).unsqueeze(-1)
        fused = spatial_maps[-1] + temporal_context
        return self.decoder(fused)