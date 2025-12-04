from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_stats import AISStats


# class ContinuousDecoderTwo(nn.Module):
#     def __init__(self, feature_dim):
#         super().__init__()
#         af = feature_dim // len(AISColDict)  # make sure this matches encoder output

#         self.mlp = nn.Linear(af, 1)  # linear layer only, no Tanh

#     def forward(self, e_xk, min_val, max_val):
#         h_n = self.mlp(e_xk)  # [b*s, 1, 1]

#         delta_range = max_val - min_val
#         safe_denominator = torch.clamp(delta_range, min=1e-6)

#         # Correct inverse normalization
#         x_hat = (h_n + 1) / 2 * safe_denominator + min_val  # map [-1,1] → [min,max]

#         # Clamp to ensure range safety
#         x_hat = torch.clamp(x_hat, min=min_val, max=max_val)

#         return x_hat  # [b, s, 1]


class ContinuousDecoderRobust(nn.Module):
    def __init__(self, feature_dim, output_dim=1):
        super().__init__()
        # If your encoder output 64 dims, this takes 64 and returns 1
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, output_dim)
        )

    def forward(self, embedding):
        """
        embedding: [batch, seq_len, feature_dim]
        """
        prediction = self.net(embedding)  # [batch, seq_len, 1]
        return prediction


class HeterogeneousAttributeDecoder(nn.Module):
    def __init__(self,
                 feature_dim,
                 stats: AISStats):
        super().__init__()

        self.stats = stats

        self.lat_decoder = ContinuousDecoderRobust(feature_dim // 2)
        self.lon_decoder = ContinuousDecoderRobust(feature_dim // 2)

    def forward(self, ais_data: torch.Tensor) -> torch.Tensor:
        b, s, f = ais_data.shape
        af = f // 2

        lat_encoding = ais_data[:, :, AISColDict.NORTHERN_DELTA.value*af: (AISColDict.NORTHERN_DELTA.value+1)*af]
        lon_encoding = ais_data[:, :, AISColDict.EASTERN_DELTA.value*af: (AISColDict.EASTERN_DELTA.value+1)*af]

        lat_hat = self.lat_decoder(lat_encoding)
        lon_hat = self.lon_decoder(lon_encoding)

        upscaled_northern_deltas = lat_hat * self.stats.std_lat + self.stats.mean_lat
        upscaled_eastern_deltas = lon_hat * self.stats.std_lon + self.stats.mean_lon

        output = torch.cat([upscaled_northern_deltas, upscaled_eastern_deltas], dim=-1)  # [b, s, num_ais_attr]

        return output
