import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelTypes.ais_stats import AISStats


class SpatioDecoder(nn.Module):
    def __init__(self, feature_dim):
        super(SpatioDecoder, self).__init__()
        self.W_lambda = nn.Linear(feature_dim * 2, 1, bias=True)
        self.W_phi = nn.Linear(feature_dim * 2, 1, bias=True)
        self.gamma_lambda = nn.Parameter(torch.tensor(0.1))
        self.gamma_phi = nn.Parameter(torch.tensor(0.1))

    def calculate_sliding_window_base(self, values, missing_value, window_size=5):
        batch_size, seq_len = values.shape
        device = values.device

        mask = (values != missing_value).float()  # [batch_size, seq_len]
        masked_values = values * mask

        window_sums = torch.zeros(batch_size, seq_len, device=device)
        window_counts = torch.zeros(batch_size, seq_len, device=device)

        pad_size = window_size
        padded_values = F.pad(masked_values, (pad_size, pad_size), "constant", 0)
        padded_mask = F.pad(mask, (pad_size, pad_size), "constant", 0)

        for i in range(2 * window_size + 1):
            window_sums += padded_values[:, i:i+seq_len]
            window_counts += padded_mask[:, i:i+seq_len]

        window_counts = torch.clamp(window_counts, min=1.0)
        window_avg = window_sums / window_counts

        return torch.deg2rad(window_avg.reshape(-1))

    def forward(self, e_lambda, e_phi, raw_lambda, raw_phi):
        # Concatenate longitude and latitude embeddings
        concat_features = torch.cat([e_lambda.squeeze(1), e_phi.squeeze(1)], dim=-1)  # [b*s, 2f]
        delta_lambda = torch.tanh(self.W_lambda(concat_features)) * self.gamma_lambda
        delta_phi = torch.tanh(self.W_phi(concat_features)) * self.gamma_phi

        with torch.no_grad():
            lambda_base = self.calculate_sliding_window_base(raw_lambda, 181)
            phi_base = self.calculate_sliding_window_base(raw_phi, 91)

        lambda_pred = lambda_base.unsqueeze(1) + delta_lambda
        phi_pred = phi_base.unsqueeze(1) + delta_phi

        spatio_pred = torch.cat([lambda_pred, phi_pred], dim=1)
        delta_coord = torch.cat([delta_lambda, delta_phi], dim=1)
        return spatio_pred, delta_coord


class CyclicalDecoder(nn.Module):

    def __init__(self, feature_dim):
        """
        Initialize the Cyclical Decoder.

        Parameters:
        - feature_dim: Dimension of the input features (f).
        - period: Period (\tau) of the cyclical variable.
        """
        super(CyclicalDecoder, self).__init__()

        # MLP with custom activation to generate trigonometric components
        self.mlp_phi = nn.Sequential(
            nn.Linear(feature_dim, 2),
            nn.Tanh(),
        )

    def forward(self, e_xk):
        """
         Forward pass for cyclical decoding.
         Parameters:
         - e_xk: Input embedding, shape [b*s, 1, f].
         Returns:
         - hat_theta: Reconstructed angular value, shape [b*s, 2].
        """
        e_xk_flat = e_xk.squeeze(1)  # [b*s, f]
        h_theta = self.mlp_phi(e_xk_flat)  # [b*s, 2]
        return h_theta


class ContinuousDecoderTwo(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # MLP to generate h_n ∈ R
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, e_xk, min_val, max_val):
        h_n = self.mlp(e_xk.squeeze(1))  # [b*s, 1]

        delta_range = max_val - min_val

        safe_denominator = torch.clamp(delta_range, min=1e-6)  # Prevent negative/zero std deviation

        x_hat = h_n * safe_denominator + min_val  # [b*s, n]

        x_hat = torch.clamp(x_hat, min=min_val, max=max_val)
        return x_hat


class DiscreteDecoder(nn.Module):

    def __init__(self, feature_dim, output_dim, num_classes, smoothing_factor=0.1):
        """
        Initialize the Discrete Decoder.

        Parameters:
        - feature_dim: Dimension of the input features (f).
        - num_classes: Number of classes (|C|).
        - class_prototypes: Predefined class prototypes (\mathbf{w}_i), shape [|C|, d].
        - smoothing_factor: Hierarchical label smoothing factor (\alpha).
        """
        super(DiscreteDecoder, self).__init__()

        # MLP to generate intermediate representation \mathbf{z}
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
        )

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=True)
        # nn.init.xavier_uniform_(self.temperature)

        # Class prototypes (\mathbf{w}_i)
        self.class_prototypes = nn.Linear(in_features=num_classes, out_features=output_dim, bias=False)  # [|C|, d]

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
        hat_y = (1 - self.smoothing_factor) * s + self.smoothing_factor * uniform_dist  # Smoothed probabilities
        # hat_y = s

        assert not torch.isnan(hat_y).any(), "discrete_decoder-hat_y contains NaN values"
        assert not torch.isinf(hat_y).any(), "discrete_decoder-hat_y contains Inf values"

        # [b*s, |C|] #[b*s, d], [b*s, d]
        return hat_y, z


class HeterogeneousAttributeDecoder(nn.Module):
    def __init__(self,
                 feature_dim,
                 stats: AISStats,
                 output_dim,
                 max_delta=300.0,
                 ):
        super().__init__()

        self.stats = stats
        self.max_delta = max_delta

        self.spatio_decoder = SpatioDecoder(feature_dim)

        self.cog_cyclical_decoder = CyclicalDecoder(feature_dim)
        self.heading_cyclical_decoder = CyclicalDecoder(feature_dim)

        self.vessel_width_continuous_decoder = ContinuousDecoderTwo(feature_dim)
        self.vessel_length_continuous_decoder = ContinuousDecoderTwo(feature_dim)
        self.vessel_draught_continuous_decoder = ContinuousDecoderTwo(feature_dim)
        self.sog_continuous_decoder = ContinuousDecoderTwo(feature_dim)
        self.rot_continuous_decoder = ContinuousDecoderTwo(feature_dim)

        self.vessel_type_discrete_decoder = DiscreteDecoder(
            feature_dim, output_dim, len(self.stats.vessel_types)+1)

    def forward(self, ais_data: torch.Tensor):
        b, s, f = ais_data.shape

        af = f // 8

        lat_encoding = ais_data[:, :, 0*af:1*af]
        lon_encoding = ais_data[:, :, 1*af:2*af]
        cog_encoding = ais_data[:, :, 2*af:3*af]
        heading_encoding = ais_data[:, :, 3*af:4*af]
        draught_encoding = ais_data[:, :, 4*af:5*af]
        sog_encoding = ais_data[:, :, 5*af:6*af]
        rot_encoding = ais_data[:, :, 6*af:7*af]
        vessel_type_encoding = ais_data[:, :, 7*af:8*af]

        draught_min = torch.tensor(self.stats.min_draught, device=ais_data.device)
        draught_max = torch.tensor(self.stats.max_draught, device=ais_data.device)
        sog_min = torch.tensor(self.stats.min_sog, device=ais_data.device)
        sog_max = torch.tensor(self.stats.max_sog, device=ais_data.device)
        rot_min = torch.tensor(self.stats.min_rot, device=ais_data.device)
        rot_max = torch.tensor(self.stats.max_rot, device=ais_data.device)

        draught_min_matrix = draught_min.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
        draught_max_matrix = draught_max.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
        sog_min_matrix = sog_min.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
        sog_max_matrix = sog_max.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
        rot_min_matrix = rot_min.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
        rot_max_matrix = rot_max.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]

        lat_hat, lon_hat = self.spatio_decoder(lat_encoding, lon_encoding)
        cog_hat = self.cog_cyclical_decoder(cog_encoding)
        heading_hat = self.heading_cyclical_decoder(heading_encoding)
        draught_hat = self.vessel_draught_continuous_decoder(
            draught_encoding, min_val=draught_min_matrix, max_val=draught_max_matrix)
        sog_hat = self.sog_continuous_decoder(sog_encoding, min_val=sog_min_matrix, max_val=sog_max_matrix)
        rot_hat = self.rot_continuous_decoder(rot_encoding, min_val=rot_min_matrix, max_val=rot_max_matrix)
        vessel_type_hat = self.vessel_type_discrete_decoder(vessel_type_encoding)

        return (lat_hat.view(b, s, -1),
                lon_hat.view(b, s, -1),
                cog_hat.view(b, s, -1),
                heading_hat.view(b, s, -1),
                draught_hat.view(b, s, -1),
                sog_hat.view(b, s, -1),
                rot_hat.view(b, s, -1),
                vessel_type_hat[0].view(b, s, -1),
                )
