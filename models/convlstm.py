import torch
import torch.nn as nn


class ConvLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config["model"]["hidden_dim"]

        self.encoder = nn.Sequential(
            nn.Conv3d(1, hidden_dim, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3,3,3), padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Conv3d(hidden_dim, 1, kernel_size=(3,3,3), padding=1)

    def forward(self, x):
        # x: (B, T, C, H, W)

        x = x.permute(0,2,1,3,4)  # (B, C, T, H, W)

        features = self.encoder(x)
        out = self.decoder(features)

        # Take last timestep
        out = out[:, :, -1, :, :]

        return out