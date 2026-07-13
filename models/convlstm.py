"""
A simple 3D CNN-based model for spatiotemporal forecasting.

This model uses a sequence of 3D convolutions to act as a simple ConvLSTM-like
encoder-decoder structure. It processes a sequence of 2D spatial maps over time.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class ConvLSTMModel(nn.Module):
    """
    A simple Conv3D model that mimics an encoder-decoder structure for
    spatiotemporal forecasting. It takes a sequence of 2D grids and predicts
    the next grid in the sequence.

    The architecture consists of two main parts:
    1. An encoder of two Conv3D layers to extract spatiotemporal features.
    2. A decoder of one Conv3D layer to project features back to the output grid.

    Args:
        config (Dict[str, Any]): A dictionary containing model configuration,
            expected to have a "model" key with "hidden_dim".
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        hidden_dim = config["model"]["hidden_dim"]

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Conv3d(in_channels=hidden_dim, out_channels=1, kernel_size=(3, 3, 3), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W), where
                B=batch, T=time steps, C=channels, H=height, W=width.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W), representing the
                predicted grid for the next time step.
        """
        # Reshape for Conv3D: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        features = self.encoder(x)
        out = self.decoder(features)

        # The decoder outputs a sequence; we take the last time step as the prediction.
        out = out[:, :, -1, :, :]

        return out