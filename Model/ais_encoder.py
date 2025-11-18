import torch
import torch.nn as nn
import math

from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_dataset_masked import AISBatch
from ModelTypes.ais_stats import AISStats


class CoordinateEncoder(nn.Module):
    """
    Encode latitude and longitude into a latent feature vector.
    Handles missing values.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # small MLP for angular embedding
        self.mlp_lat = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.mlp_lon = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, lon, lat):
        """
        lon, lat: [b, s, 1] in degrees
        returns: [b, s, d_model]
        """
        # missing values mask
        lon_mask = (lon == 181.0)
        lat_mask = (lat == 91.0)

        # convert degrees → radians
        lon_rad = torch.deg2rad(lon)
        lat_rad = torch.deg2rad(lat)

        # sin/cos embedding
        lat_features = torch.cat([
            torch.sin(lat_rad),
            torch.cos(lat_rad)
        ], dim=-1)  # [b, s, 2]

        lon_features = torch.cat([
            torch.sin(lon_rad),
            torch.cos(lon_rad)
        ], dim=-1)  # [b, s, 2]

        encoded_lat = self.mlp_lat(lat_features)  # [b, s, d_model]
        encoded_lon = self.mlp_lon(lon_features)  # [b, s, d_model]

        # zero out missing positions
        encoded_lat = encoded_lat.masked_fill(lat_mask, 0.0).unsqueeze(2)
        encoded_lon = encoded_lon.masked_fill(lon_mask, 0.0).unsqueeze(2)

        encoded = torch.cat([encoded_lon, encoded_lat], dim=2)

        return encoded


class CyclicalEncoder(nn.Module):
    """
    Cyclical Encoder for attributes with physical periodicity (e.g., heading angle or course over ground).
    Implements the formula:
        e_p = phi(W_p * [sin(pi * x_p / 180), cos(pi * x_p / 180)]^T + b_p)
    where phi(x) = x + sin^2(x) introduces explicit periodic inductive bias.
    """

    def __init__(self, d_model_E):
        """
        Initialize the encoder.

        Args:
            d_model_E (int): Dimensionality of the output embedding.
        """
        super().__init__()
        # Linear layer to map trigonometric features to d_model_E dimensions
        self.linear = nn.Linear(2, d_model_E)
        # Activation function: phi(x) = x + sin^2(x)
        # self.activation = lambda x: x + torch.sin(x) ** 2

    def forward(self, x):
        mask = (x == -1)

        # Step 1: Convert input angles from degrees to radians
        x_rad = torch.deg2rad(x) - torch.pi  # Shape [b, s, 1]

        # Step 2: Compute sine and cosine features
        sin_features = torch.sin(x_rad)  # Shape [b, s, 1]
        cos_features = torch.cos(x_rad)  # Shape [b, s, 1]

        # Step 3: Combine sine and cosine features into a single vector [b, s, 2]
        trig_features = torch.cat([sin_features, cos_features], dim=-1)  # Shape [b, s, 2]

        # Step 4: Apply linear transformation and activation
        x_encoded = self.linear(trig_features)  # Shape [b, s, d_model_E]
        # x_encoded = self.activation(x_encoded)  # Shape [b, s, d_model_E]

        x_encoded = x_encoded.masked_fill(mask.expand_as(x_encoded), 0.0)

        # Step 5: Reshape to [b, s, 1, f]
        output = x_encoded.unsqueeze(2)  # Shape [b, s, 1, f]

        return output


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


class DiscreteEncoder(nn.Module):
    """Discrete Feature Encoder with missing value handling"""

    def __init__(self, d_model_E, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, d_model_E)

    def forward(self, x):
        x = x.long().squeeze(-1)  # [b, s]
        out = self.embedding(x)  # [b, s, d_model_E]
        return out.unsqueeze(2)  # [b, s, 1, d_model_E]


class HeterogeneousAttributeEncoder(nn.Module):

    def __init__(self, feature_dim, stats: AISStats):
        super().__init__()
        self.stats = stats

        self.coordinate_encoder = CoordinateEncoder(feature_dim)
        self.cog_cyclical_encoder = CyclicalEncoder(feature_dim)
        self.heading_angle_cyclical_encoder = CyclicalEncoder(feature_dim)
        self.vessel_draught_continuous_encoder = ContinuousEncoderTwo(feature_dim)
        self.sog_continuous_encoder = ContinuousEncoderTwo(feature_dim)
        self.rot_continuous_encoder = ContinuousEncoderTwo(feature_dim)
        self.vessel_type_discrete_encoder = DiscreteEncoder(
            feature_dim, num_classes=len(stats.vessel_types))

        self.output_dim = (len(AISColDict) * feature_dim)
        self.latitude_col_idx = AISColDict.LATITUDE.value

    def forward(self, ais_data: torch.Tensor):
        tensor_input = ais_data
        b, s, _ = ais_data.shape  # [b, s, n]

        draught_min = torch.tensor(self.stats.min_draught, device=tensor_input.device)
        draught_max = torch.tensor(self.stats.max_draught, device=tensor_input.device)
        sog_min = torch.tensor(self.stats.min_sog, device=tensor_input.device)
        sog_max = torch.tensor(self.stats.max_sog, device=tensor_input.device)
        rot_min = torch.tensor(self.stats.min_rot, device=tensor_input.device)
        rot_max = torch.tensor(self.stats.max_rot, device=tensor_input.device)
        draught_min_matrix = draught_min.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
        draught_max_matrix = draught_max.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
        sog_min_matrix = sog_min.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
        sog_max_matrix = sog_max.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
        rot_min_matrix = rot_min.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
        rot_max_matrix = rot_max.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]

        spatial_output = self.coordinate_encoder(
            tensor_input[:, :, AISColDict.LONGITUDE.value:AISColDict.LONGITUDE.value+1],
            tensor_input[:, :, AISColDict.LATITUDE.value:AISColDict.LATITUDE.value+1]
        )

        cog_output = self.cog_cyclical_encoder(
            tensor_input[:, :, AISColDict.COG.value:AISColDict.COG.value+1])

        heading_angle_output = self.heading_angle_cyclical_encoder(
            tensor_input[:, :, AISColDict.HEADING.value:AISColDict.HEADING.value+1])

        vessel_type_output = self.vessel_type_discrete_encoder(
            tensor_input[:, :, AISColDict.VESSEL_TYPE.value:AISColDict.VESSEL_TYPE.value+1])

        draught_output = self.vessel_draught_continuous_encoder(
            tensor_input[:, :, AISColDict.DRAUGHT.value:AISColDict.DRAUGHT.value+1],
            min_val=draught_min_matrix,
            max_val=draught_max_matrix)

        sog_output = self.sog_continuous_encoder(
            tensor_input[:, :, AISColDict.SOG.value:AISColDict.SOG.value+1],
            min_val=sog_min_matrix,
            max_val=sog_max_matrix)

        rot_output = self.rot_continuous_encoder(
            tensor_input[:, :, AISColDict.ROT.value:AISColDict.ROT.value+1],
            min_val=rot_min_matrix,
            max_val=rot_max_matrix)

        output = torch.cat((spatial_output,
                            rot_output, sog_output, cog_output, heading_angle_output,
                            vessel_type_output, draught_output
                            ),
                           dim=2)  # shape [b, s, len(AISColDict), feature_dim]

        output = output.view(b, s, -1)  # shape [b, s, len(AISColDict)*feature_dim]

        return output
