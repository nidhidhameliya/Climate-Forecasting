import torch
import torch.nn as nn


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config["model"]["hidden_dim"]

        self.input_projection = nn.Conv2d(1, d_model, kernel_size=1)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )

        self.decoder = nn.Conv2d(d_model, 1, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        feature_maps = [self.input_projection(x[:, t]) for t in range(T)]
        sequence = torch.stack(
            [feature_map.mean(dim=(2, 3)) for feature_map in feature_maps], dim=1
        )
        temporal_context = self.transformer(sequence)[:, -1]
        fused = feature_maps[-1] + temporal_context.unsqueeze(-1).unsqueeze(-1)
        return self.decoder(fused)