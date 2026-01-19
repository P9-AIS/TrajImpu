from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_stats import AISStats

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

        self.northern_decoder = ContinuousDecoderRobust(feature_dim // 2)
        self.eastern_decoder = ContinuousDecoderRobust(feature_dim // 2)

    def forward(self, ais_data: torch.Tensor) -> torch.Tensor:
        b, s, f = ais_data.shape
        af = f // 2

        northern_encoding = ais_data[:, :, AISColDict.NORTHERN_DELTA.value*af: (AISColDict.NORTHERN_DELTA.value+1)*af]
        eastern_encoding = ais_data[:, :, AISColDict.EASTERN_DELTA.value*af: (AISColDict.EASTERN_DELTA.value+1)*af]

        northern_hat = self.northern_decoder(northern_encoding)
        eastern_hat = self.eastern_decoder(eastern_encoding)

        upscaled_northern_deltas = northern_hat * self.stats.std_delta_n + self.stats.mean_abs_delta_n
        upscaled_eastern_deltas = eastern_hat * self.stats.std_delta_e + self.stats.mean_abs_delta_e

        output = torch.cat([upscaled_northern_deltas, upscaled_eastern_deltas], dim=-1)  # [b, s, num_ais_attr]

        return output
