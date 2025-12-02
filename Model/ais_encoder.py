import torch
import torch.nn as nn
from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_stats import AISStats


class ContinuousEncoderTwo(nn.Module):
    def __init__(self, d_model_E):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model_E),
            nn.Tanh(),
        )

    def forward(self, x, min_val, max_val):
        missing_mask = torch.eq(x, -1)  # [b, s, 1]

        denominator = torch.clamp(max_val - min_val, min=1e-6)
        x_norm = 2 * (x - min_val) / denominator - 1

        batch_size, seq_len, _ = x_norm.shape

        x_encoded = self.mlp(x_norm.view(-1, 1))  # [b*s, d_model_E]
        x_encoded = x_encoded.view(batch_size, seq_len, 1, -1)  # [b, s, 1, d_model_E]

        x_encoded = x_encoded.masked_fill(missing_mask.unsqueeze(-1), 0.0)

        return x_encoded


class HeterogeneousAttributeEncoder(nn.Module):

    def __init__(self, feature_dim, stats: AISStats):
        super().__init__()
        self.stats = stats

        self.lat_continous_encoder = ContinuousEncoderTwo(feature_dim)
        self.lon_continous_encoder = ContinuousEncoderTwo(feature_dim)

        self.output_dim = (len(AISColDict) * feature_dim)
        self.latitude_col_idx = AISColDict.NORTHERN_DELTA.value

    def forward(self, ais_data: torch.Tensor):
        tensor_input = ais_data
        b, s, _ = ais_data.shape  # [b, s, n]

        lat_min = torch.tensor(self.stats.min_lat, device=tensor_input.device)
        lat_max = torch.tensor(self.stats.max_lat, device=tensor_input.device)
        lon_min = torch.tensor(self.stats.min_lon, device=tensor_input.device)
        lon_max = torch.tensor(self.stats.max_lon, device=tensor_input.device)

        lat_min_matrix = lat_min.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
        lat_max_matrix = lat_max.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
        lon_min_matrix = lon_min.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
        lon_max_matrix = lon_max.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)

        lat_output = self.lat_continous_encoder(
            tensor_input[:, :, AISColDict.NORTHERN_DELTA.value:AISColDict.NORTHERN_DELTA.value+1],
            lat_min_matrix, lat_max_matrix)

        lon_output = self.lon_continous_encoder(
            tensor_input[:, :, AISColDict.EASTERN_DELTA.value:AISColDict.EASTERN_DELTA.value+1],
            lon_min_matrix, lon_max_matrix)

        output = torch.cat((lat_output, lon_output), dim=2)  # shape [b, s, len(AISColDict), feature_dim]
        output = output.view(b, s, -1)  # shape [b, s, len(AISColDict)*feature_dim]

        return output
