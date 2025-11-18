from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_stats import AISStats


class CyclicalDecoder(nn.Module):

    def __init__(self, feature_dim):
        """
        Initialize the Cyclical Decoder.

        Parameters:
        - feature_dim: Dimension of the input features (f).
        - period: Period (\tau) of the cyclical variable.
        """
        super(CyclicalDecoder, self).__init__()

        af = feature_dim // len(AISColDict)

        # MLP with custom activation to generate trigonometric components
        self.mlp_phi = nn.Sequential(
            nn.Linear(af, 2),
            nn.Tanh(),
        )

    def forward(self, e_xk):
        b, s, _ = e_xk.shape
        e_xk_flat = e_xk.squeeze(1)  # [b*s, f]
        h_theta = self.mlp_phi(e_xk_flat)  # [b, s, 2]

        hat_theta = torch.atan2(h_theta[:, :, 0], h_theta[:, :, 1])  # radians
        hat_theta_deg = (torch.rad2deg(hat_theta) % 360)
        hat_theta_deg = hat_theta_deg.view(b, s, 1)

        return hat_theta_deg


class ContinuousDecoderTwo(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        af = feature_dim // len(AISColDict)  # make sure this matches encoder output

        self.mlp = nn.Linear(af, 1)  # linear layer only, no Tanh

    def forward(self, e_xk, min_val, max_val):
        h_n = self.mlp(e_xk.squeeze(1))  # [B*S, 1]

        delta_range = max_val - min_val
        safe_denominator = torch.clamp(delta_range, min=1e-6)

        # Correct inverse normalization
        x_hat = (h_n + 1) / 2 * safe_denominator + min_val  # map [-1,1] → [min,max]

        # Clamp to ensure range safety
        x_hat = torch.clamp(x_hat, min=min_val, max=max_val)

        return x_hat


class DiscreteDecoder(nn.Module):

    def __init__(self, feature_dim, output_dim, num_classes, smoothing_factor=0.1):
        """
        Initialize the Discrete Decoder.

        Parameters:
        - feature_dim: Dimension of the input features (f).
        - num_classes: Number of classes (|C|).
        - class_prototypes: Predefined class prototypes (mathbf{w}_i), shape [|C|, d].
        - smoothing_factor: Hierarchical label smoothing factor (\alpha).
        """
        super(DiscreteDecoder, self).__init__()

        af = feature_dim // len(AISColDict)

        # MLP to generate intermediate representation \mathbf{z}
        self.mlp = nn.Sequential(
            nn.Linear(af, af),
        )

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=True)
        # nn.init.xavier_uniform_(self.temperature)

        # Class prototypes (\mathbf{w}_i)
        self.class_prototypes = nn.Linear(in_features=num_classes, out_features=af, bias=False)  # [|C|, d]

        # Smoothing factor (\alpha)
        self.smoothing_factor = smoothing_factor

    def forward(self, e_xk):
        """
        Forward pass for discrete decoding.

        Parameters:
        - e_xk: Input embedding, shape [b*s, 1, f].

        Returns:
        - hat_y: Class probability distribution, shape [b*s, |C|].
        """
        # Step 1: Generate intermediate representation \mathbf{z}
        z = self.mlp(e_xk.squeeze(1))  # [b*s, d]

        # Step 2: Compute similarity scores with class prototypes
        logits = torch.matmul(z, self.class_prototypes.weight) / self.temperature  # [b*s, |C|]

        # Step 3: Apply softmax to compute class probabilities
        s = F.softmax(logits, dim=-1)  # [b*s, |C|]

        # # Step 4: Hierarchical label smoothing
        uniform_dist = torch.full_like(s, 1.0 / s.size(-1))  # Uniform distribution [|C|]
        dist_y = (1 - self.smoothing_factor) * s + self.smoothing_factor * uniform_dist  # Smoothed probabilities
        vessel_type_hat = torch.argmax(dist_y, dim=-1).unsqueeze(-1)

        # get the final

        assert not torch.isnan(dist_y).any(), "discrete_decoder-hat_y contains NaN values"
        assert not torch.isinf(dist_y).any(), "discrete_decoder-hat_y contains Inf values"

        # [b*s, |C|] #[b*s, d], [b*s, d]
        return vessel_type_hat, logits


@dataclass
class ExtraDecodeOutput:
    vessel_type_prob_logits: torch.Tensor


class HeterogeneousAttributeDecoder(nn.Module):
    def __init__(self,
                 feature_dim,
                 stats: AISStats,
                 output_dim,
                 ):
        super().__init__()

        self.stats = stats

        self.lat_decoder = ContinuousDecoderTwo(feature_dim)
        self.lon_decoder = ContinuousDecoderTwo(feature_dim)

        self.cog_cyclical_decoder = CyclicalDecoder(feature_dim)
        self.heading_cyclical_decoder = CyclicalDecoder(feature_dim)

        self.vessel_draught_continuous_decoder = ContinuousDecoderTwo(feature_dim)
        self.sog_continuous_decoder = ContinuousDecoderTwo(feature_dim)
        self.rot_continuous_decoder = ContinuousDecoderTwo(feature_dim)

        self.vessel_type_discrete_decoder = DiscreteDecoder(
            feature_dim, output_dim, len(self.stats.vessel_types)+1)

    def forward(self, ais_data: torch.Tensor):
        b, s, f = ais_data.shape

        af = f // len(AISColDict)

        lat_encoding = ais_data[:, :, AISColDict.LATITUDE.value*af: (AISColDict.LATITUDE.value+1)*af]
        lon_encoding = ais_data[:, :, AISColDict.LONGITUDE.value*af: (AISColDict.LONGITUDE.value+1)*af]
        cog_encoding = ais_data[:, :, AISColDict.COG.value*af: (AISColDict.COG.value+1)*af]
        heading_encoding = ais_data[:, :, AISColDict.HEADING.value*af: (AISColDict.HEADING.value+1)*af]
        draught_encoding = ais_data[:, :, AISColDict.DRAUGHT.value*af: (AISColDict.DRAUGHT.value+1)*af]
        sog_encoding = ais_data[:, :, AISColDict.SOG.value*af: (AISColDict.SOG.value+1)*af]
        rot_encoding = ais_data[:, :, AISColDict.ROT.value*af: (AISColDict.ROT.value+1)*af]
        vessel_type_encoding = ais_data[:, :, AISColDict.VESSEL_TYPE.value*af: (AISColDict.VESSEL_TYPE.value+1)*af]

        lat_min = torch.tensor(self.stats.min_lat, device=ais_data.device)
        lat_max = torch.tensor(self.stats.max_lat, device=ais_data.device)
        lon_min = torch.tensor(self.stats.min_lon, device=ais_data.device)
        lon_max = torch.tensor(self.stats.max_lon, device=ais_data.device)
        draught_min = torch.tensor(self.stats.min_draught, device=ais_data.device)
        draught_max = torch.tensor(self.stats.max_draught, device=ais_data.device)
        sog_min = torch.tensor(self.stats.min_sog, device=ais_data.device)
        sog_max = torch.tensor(self.stats.max_sog, device=ais_data.device)
        rot_min = torch.tensor(self.stats.min_rot, device=ais_data.device)
        rot_max = torch.tensor(self.stats.max_rot, device=ais_data.device)

        lat_min_matrix = lat_min.view(1, 1, 1).expand(b, s, 1)
        lat_max_matrix = lat_max.view(1, 1, 1).expand(b, s, 1)
        lon_min_matrix = lon_min.view(1, 1, 1).expand(b, s, 1)
        lon_max_matrix = lon_max.view(1, 1, 1).expand(b, s, 1)
        draught_min_matrix = draught_min.view(1, 1, 1).expand(b, s, 1)
        draught_max_matrix = draught_max.view(1, 1, 1).expand(b, s, 1)
        sog_min_matrix = sog_min.view(1, 1, 1).expand(b, s, 1)
        sog_max_matrix = sog_max.view(1, 1, 1).expand(b, s, 1)
        rot_min_matrix = rot_min.view(1, 1, 1).expand(b, s, 1)
        rot_max_matrix = rot_max.view(1, 1, 1).expand(b, s, 1)

        lat_hat = self.lat_decoder(lat_encoding, lat_min_matrix, lat_max_matrix)
        lon_hat = self.lon_decoder(lon_encoding, lon_min_matrix, lon_max_matrix)
        cog_hat = self.cog_cyclical_decoder(cog_encoding)
        heading_hat = self.heading_cyclical_decoder(heading_encoding)
        draught_hat = self.vessel_draught_continuous_decoder(
            draught_encoding, min_val=draught_min_matrix, max_val=draught_max_matrix)
        sog_hat = self.sog_continuous_decoder(sog_encoding, min_val=sog_min_matrix, max_val=sog_max_matrix)
        rot_hat = self.rot_continuous_decoder(rot_encoding, min_val=rot_min_matrix, max_val=rot_max_matrix)
        vessel_type_hat, prob_dist = self.vessel_type_discrete_decoder(vessel_type_encoding)

        output = torch.cat([
            lat_hat, lon_hat, rot_hat, sog_hat, cog_hat, heading_hat, vessel_type_hat, draught_hat
        ], dim=-1).view(b, s, -1)  # [b, s, num_ais_attr]

        extra = ExtraDecodeOutput(
            vessel_type_prob_logits=prob_dist.view(b, s, -1)
        )

        return output, extra
