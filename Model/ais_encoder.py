import torch
import torch.nn as nn
import math

from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_dataset_masked import AISBatch
from ModelTypes.ais_stats import AISStats


class CoordinateEncoder(nn.Module):
    """
    Spatial Coordinate Encoder using hybrid spherical-harmonic encoding.
    Inputs: lon (longitude) and lat (latitude) with shape [b, s, 1].
    Outputs: Encoded features with shape [b, s, 2, f], where f = d_model_E.
    """

    def __init__(self, d_model_E):
        super().__init__()
        self.d_model_E = d_model_E

       # Learnable affine transformation parameters for each dimension
        self.linear_lon = nn.Linear(9, d_model_E)  # For longitude
        self.linear_lat = nn.Linear(9, d_model_E)  # For latitude
        self.activation = nn.Tanh()  # Hyperbolic tangent activation function

        # Initialize weights and biases using Xavier initialization
        nn.init.xavier_uniform_(self.linear_lon.weight)
        nn.init.zeros_(self.linear_lon.bias)
        nn.init.xavier_uniform_(self.linear_lat.weight)
        nn.init.zeros_(self.linear_lat.bias)

    def forward(self, lon, lat):
        """
        Args:
            lon (torch.Tensor): Longitude tensor of shape [b, s, 1] (in degrees).
            lat (torch.Tensor): Latitude tensor of shape [b, s, 1] (in degrees).
        Returns:
            torch.Tensor: Encoded features of shape [b, s, 2, f], where f = d_model_E.
        """
        lon_missing_mask = torch.isclose(lon, torch.tensor(181.0, dtype=torch.float32), atol=1e-5)  # [b, s, 1]
        lat_missing_mask = torch.isclose(lat, torch.tensor(91.0, dtype=torch.float32), atol=1e-5)  # [b, s, 1]
        # Convert degrees to radians
        lon_rad = torch.deg2rad(lon)  # Shape [b, s, 1]
        lat_rad = torch.deg2rad(lat)  # Shape [b, s, 1]

        # Compute the hybrid spherical-harmonic feature vector
        sin_lon = torch.sin(lon_rad)
        cos_lon = torch.cos(lon_rad)
        sin_lat = torch.sin(lat_rad)
        cos_lat = torch.cos(lat_rad)

        # Feature vector components
        f1 = sin_lon * cos_lat  # sin(λ)cos(φ)
        f2 = cos_lon * cos_lat  # cos(λ)cos(φ)
        f3 = sin_lat            # sin(φ)
        f4 = torch.sin(2 * lon_rad) * cos_lat  # sin(2λ)cos(φ)
        f5 = torch.cos(2 * lon_rad) * cos_lat  # cos(2λ)cos(φ)

        # Combine into a single feature vector [b, s, 9]
        features = torch.cat([sin_lon, cos_lon, sin_lat, cos_lat, f1, f2, f3, f4, f5], dim=-1)  # Shape [b, s, 9]

        # Apply learnable affine transformations and activation for each dimension
        encoded_lon = self.activation(self.linear_lon(features))  # Shape [b, s, d_model_E]
        encoded_lat = self.activation(self.linear_lat(features))  # Shape [b, s, d_model_E]
        encoded_lon = encoded_lon.masked_fill(lon_missing_mask.expand_as(encoded_lon), 0.0)
        encoded_lat = encoded_lat.masked_fill(lat_missing_mask.expand_as(encoded_lat), 0.0)

        # Stack the encoded features along the feature dimension
        output = torch.stack([encoded_lat, encoded_lon], dim=2)  # Shape [b, s, 2, d_model_E]

        return output


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
        x_norm = (x - min_val) / denominator

        batch_size, seq_len, _ = x_norm.shape

        x_encoded = self.mlp(x_norm.view(-1, 1))  # [b*s, d_model_E]
        x_encoded = x_encoded.view(batch_size, seq_len, 1, -1)  # [b, s, 1, d_model_E]

        x_encoded = x_encoded.masked_fill(missing_mask.unsqueeze(-1), 0.0)

        return x_encoded


class DiscreteEncoder(nn.Module):
    """Discrete Feature Encoder with missing value handling"""

    def __init__(self, d_model_E, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, d_model_E, padding_idx=num_classes)

        with torch.no_grad():
            self.embedding.weight[num_classes].zero_()

    def forward(self, x):
        x = x.long().squeeze(-1)  # [b, s]
        out = self.embedding(x)  # [b, s, d_model_E]
        return out.unsqueeze(2)  # [b, s, 1, d_model_E]


class HeterogeneousAttributeEncoder(nn.Module):

    def __init__(self, feature_dim,
                 stats: AISStats,
                 max_delta: float):
        super().__init__()
        self.stats = stats
        self.max_delta = max_delta

        self.coordinate_encoder = CoordinateEncoder(feature_dim)
        self.cog_cyclical_encoder = CyclicalEncoder(feature_dim)
        self.heading_angle_cyclical_encoder = CyclicalEncoder(feature_dim)
        self.vessel_draught_continuous_encoder = ContinuousEncoderTwo(feature_dim)
        self.sog_continuous_encoder = ContinuousEncoderTwo(feature_dim)
        self.rot_continuous_encoder = ContinuousEncoderTwo(feature_dim)
        self.vessel_type_discrete_encoder = DiscreteEncoder(
            feature_dim, num_classes=max(int(t) for t in stats.vessel_types))

        self.output_dim = (8 * feature_dim)

        self.timestamp_col_idx = AISColDict.TIMESTAMP.value
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
                            cog_output, heading_angle_output,
                            draught_output, sog_output, rot_output,
                            vessel_type_output),
                           dim=2)  # shape [b, s, 8, feature_dim]

        output = output.view(b, s, -1)  # shape [b, s, 8*feature_dim]

        return output
