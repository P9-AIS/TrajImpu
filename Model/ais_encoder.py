import torch
import torch.nn as nn
from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_stats import AISStats


# class ContinuousEncoderTwo(nn.Module):
#     def __init__(self, d_model_E):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(1, d_model_E),
#             nn.Tanh(),
#         )

#     def forward(self, x, min_val, max_val):
#         denominator = torch.clamp(max_val - min_val, min=1e-6)
#         x_norm = 2 * (x - min_val) / denominator - 1

#         batch_size, seq_len, _ = x_norm.shape

#         x_encoded = self.mlp(x_norm.view(-1, 1))  # [b*s, d_model_E]
#         x_encoded = x_encoded.view(batch_size, seq_len, 1, -1)  # [b, s, 1, d_model_E]

#         return x_encoded


import torch
import torch.nn as nn
import numpy as np


class ContinuousDeltaEncoder(nn.Module):
    def __init__(self, d_model_E):
        super().__init__()
        # We split the output dimension: half for sin, half for cos
        self.d_half = d_model_E // 2

        # Learnable scaling factor (frequency)
        # Initializes with a distribution that covers different scales of movement
        self.freq_weights = nn.Parameter(torch.randn(1, self.d_half) * 0.1)

        # Final mixing layer to combine the frequencies
        self.out_proj = nn.Sequential(
            nn.Linear(2 * self.d_half, d_model_E),
            nn.GELU()
        )

    def forward(self, x):
        """
        x: [batch, seq_len, 1] - Delta values
        mask: [batch, seq_len] - True if data is MISSING (optional)
        """

        # Replace NaNs with 0 temporarily so math doesn't crash
        x_clean = torch.nan_to_num(x, nan=0.0)

        # 2. Fourier Feature Encoding (Sinusoidal)
        # Project scalar x to vector: [batch, seq, 1] @ [1, d_half] -> [batch, seq, d_half]
        # This creates 'frequencies' for the movement
        projected = x_clean @ self.freq_weights

        # Create sin and cos features
        # Concatenate -> [batch, seq, d_model]
        x_embedding = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

        # 3. Mix features
        x_encoded = self.out_proj(x_embedding)

        # Reshape to your desired output: [b, s, 1, d]
        return x_encoded.unsqueeze(2)


class HeterogeneousAttributeEncoder(nn.Module):

    def __init__(self, feature_dim, stats: AISStats):
        super().__init__()
        self.stats = stats

        self.lat_continous_encoder = ContinuousDeltaEncoder(feature_dim)
        self.lon_continous_encoder = ContinuousDeltaEncoder(feature_dim)

        self.output_dim = (len(AISColDict) * feature_dim)
        self.latitude_col_idx = AISColDict.NORTHERN_DELTA.value

    def forward(self, ais_data: torch.Tensor):
        tensor_input = ais_data
        b, s, _ = ais_data.shape  # [b, s, n]

        northern_deltas = tensor_input[:, :, AISColDict.NORTHERN_DELTA.value:AISColDict.NORTHERN_DELTA.value+1]
        eastern_deltas = tensor_input[:, :, AISColDict.EASTERN_DELTA.value:AISColDict.EASTERN_DELTA.value+1]

        scaled_northern_deltas = (northern_deltas - self.stats.mean_lat) / self.stats.std_lat
        scaled_eastern_deltas = (eastern_deltas - self.stats.mean_lon) / self.stats.std_lon

        lat_output = self.lat_continous_encoder(scaled_northern_deltas)
        lon_output = self.lon_continous_encoder(scaled_eastern_deltas)

        output = torch.cat((lat_output, lon_output), dim=2)  # shape [b, s, len(AISColDict), feature_dim]
        output = output.view(b, s, -1)  # shape [b, s, len(AISColDict)*feature_dim]

        return output
