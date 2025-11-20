import torch
import torch.nn as nn

from ForceProviders.i_force_provider import IForceProvider
from ModelTypes.ais_dataset_masked import AISBatch


class AFAModule(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int,  *force_providers: IForceProvider):
        super().__init__()
        self._feature_dim = feature_dim
        self._num_heads = num_heads
        self._force_providers = force_providers

        # Multihead cross-attention: AIS queries attend to forces
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Linear projection for forces to feature space (for K/V)
        self.force_proj = nn.Linear(3, feature_dim)

    def forward(self, lats: torch.Tensor, lons: torch.Tensor, encoded_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, feature_dim = encoded_data.shape

        F_set = self._get_forces(lats, lons)  # [b, s, num_forces, 3]
        num_forces = F_set.shape[2]

        # Flatten batch/time dims for attention
        Q = encoded_data.view(b * s, 1, feature_dim)  # [b*s, 1, feature_dim]
        K = self.force_proj(F_set.view(b * s, num_forces, 3))  # [b*s, num_forces, feature_dim]
        V = K

        # Cross-attention: AIS queries attend to forces
        attn_output, attn_weights = self.cross_attn(Q, K, V)
        # attn_output: [b*s, 1, feature_dim], attn_weights: [b*s, 1, num_forces]

        attn_output = attn_output.squeeze(1).view(b, s, feature_dim)  # [b, s, feature_dim]
        attn_scores = attn_weights.squeeze(1).view(b, s, num_forces)  # [b, s, num_forces]

        return attn_output, attn_scores

    def _get_forces(self, lats: torch.Tensor, lons: torch.Tensor) -> torch.Tensor:
        b, s = lats.shape
        num_forces = len(self._force_providers)
        lat_lons = torch.stack((lats, lons), dim=-1)  # [b, s, 2]

        all_forces = torch.empty(b, s, num_forces, 3, device=lats.device)
        for i, provider in enumerate(self._force_providers):
            forces = provider.get_forces(lat_lons)
            all_forces[:, :, i, :] = forces

        return all_forces
