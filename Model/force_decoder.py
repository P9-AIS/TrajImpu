import torch
import torch.nn as nn


class ForceDecoder(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.decode_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 2),
            nn.Tanh()  # Brings components to [-1, 1]
        )

    def forward(self, force_embedding: torch.Tensor) -> torch.Tensor:
        forces = self.decode_proj(force_embedding)

        # Calculate magnitude
        # Add epsilon to avoid division by zero
        magnitudes = torch.norm(forces, dim=-1, keepdim=True) + 1e-6

        # If magnitude > 1, scale it down to 1. If < 1, leave it alone.
        # This preserves direction while capping intensity.
        scale_factor = torch.clamp(magnitudes, max=1.0) / magnitudes

        return forces * scale_factor
