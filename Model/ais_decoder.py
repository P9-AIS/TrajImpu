from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_stats import AISStats


class ContinuousDecoderTwo(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        af = feature_dim // len(AISColDict)  # make sure this matches encoder output

        self.mlp = nn.Linear(af, 1)  # linear layer only, no Tanh

    def forward(self, e_xk, min_val, max_val):
        h_n = self.mlp(e_xk)  # [b*s, 1, 1]

        delta_range = max_val - min_val
        safe_denominator = torch.clamp(delta_range, min=1e-6)

        # Correct inverse normalization
        x_hat = (h_n + 1) / 2 * safe_denominator + min_val  # map [-1,1] → [min,max]

        # Clamp to ensure range safety
        x_hat = torch.clamp(x_hat, min=min_val, max=max_val)

        return x_hat  # [b, s, 1]


class HeterogeneousAttributeDecoder(nn.Module):
    def __init__(self,
                 feature_dim,
                 stats: AISStats):
        super().__init__()

        self.stats = stats

        self.lat_decoder = ContinuousDecoderTwo(feature_dim)
        self.lon_decoder = ContinuousDecoderTwo(feature_dim)

    def forward(self, ais_data: torch.Tensor) -> torch.Tensor:
        b, s, f = ais_data.shape

        af = f // len(AISColDict)

        lat_encoding = ais_data[:, :, AISColDict.NORTHERN_DELTA.value*af: (AISColDict.NORTHERN_DELTA.value+1)*af]
        lon_encoding = ais_data[:, :, AISColDict.EASTERN_DELTA.value*af: (AISColDict.EASTERN_DELTA.value+1)*af]

        lat_min = torch.tensor(self.stats.min_lat, device=ais_data.device)
        lat_max = torch.tensor(self.stats.max_lat, device=ais_data.device)
        lon_min = torch.tensor(self.stats.min_lon, device=ais_data.device)
        lon_max = torch.tensor(self.stats.max_lon, device=ais_data.device)

        lat_min_matrix = lat_min.view(1, 1, 1).expand(b, s, 1)
        lat_max_matrix = lat_max.view(1, 1, 1).expand(b, s, 1)
        lon_min_matrix = lon_min.view(1, 1, 1).expand(b, s, 1)
        lon_max_matrix = lon_max.view(1, 1, 1).expand(b, s, 1)

        lat_hat = self.lat_decoder(lat_encoding, lat_min_matrix, lat_max_matrix)
        lon_hat = self.lon_decoder(lon_encoding, lon_min_matrix, lon_max_matrix)

        output = torch.cat([lat_hat, lon_hat], dim=-1)  # [b, s, num_ais_attr]

        return output
