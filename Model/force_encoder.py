import torch
import torch.nn as nn

from ForceProviders.i_force_provider import IForceProvider
from ForceUtils.geo_converter import GeoConverter as gc


class ForceEncoder(nn.Module):
    def __init__(self, feature_dim: int, force_provider: IForceProvider):
        super().__init__()
        self._feature_dim = feature_dim
        self._force_provider = force_provider

        self.force_proj = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, northerns: torch.Tensor, easterns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_forces = self.get_raw_forces(northerns, easterns)  # [b, s, 2]
        force_embedding = self.force_proj(raw_forces)  # [b, s, feature_dim]
        return force_embedding, raw_forces

    def get_raw_forces(self, northerns: torch.Tensor, easterns: torch.Tensor) -> torch.Tensor:
        raw_forces = self._force_provider.get_forces_tensor(northerns, easterns).to(northerns.device)  # [b, s, 2]
        return raw_forces
