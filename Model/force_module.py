import torch
import torch.nn as nn

from ForceProviders.i_force_provider import IForceProvider


class ForceModule(nn.Module):
    def __init__(self, feature_dim: int, force_provider: IForceProvider):
        super().__init__()
        self._feature_dim = feature_dim
        self._force_provider = force_provider

        self.force_proj = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, lats: torch.Tensor, lons: torch.Tensor, encoded_data: torch.Tensor) -> torch.Tensor:
        lat_lons = torch.stack((lats, lons), dim=-1)  # [b, s, 2]
        raw_forces = self._force_provider.get_forces_tensor(lat_lons).to(encoded_data.device)  # [b, s, 2]

        force_embedding = self.force_proj(raw_forces)  # [b, s, feature_dim]
        combined = torch.cat((encoded_data, force_embedding), dim=-1)

        return combined
