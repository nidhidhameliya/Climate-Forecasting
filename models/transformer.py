import torch
import torch.nn as nn


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config["model"]["hidden_dim"]

        self.flatten = nn.Flatten(2)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4),
            num_layers=2
        )

        self.proj = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B, T, -1)
        x = self.proj(x)

        x = x.permute(1,0,2)

        out = self.transformer(x)

        out = out[-1]

        out = self.output(out)

        return out.view(B,1,1,1)