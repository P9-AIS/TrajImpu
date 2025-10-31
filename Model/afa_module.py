import torch
import torch.nn as nn


class AFAModule(nn.Module):
    def __init__(self, feature_dim, num_heads, force_provider):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.force_provider = force_provider

        # Multihead cross-attention: AIS queries attend to forces
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Linear projection for forces to feature space (for K/V)
        self.force_proj = nn.Linear(3, feature_dim)
        # Project attention output back to force space (x,y,z)
        self.output_proj = nn.Linear(feature_dim, 3)

    def forward(self, ais_data, encoded_data):
        """
        Args:
            ais_data: [b, s, num_attr]
            encoded_data: [b, s, num_attr, feature_dim]
        Returns:
            aggregated_forces: [b, s, 3]
            attn_scores: [b, s, num_attr, num_forces]
        """
        b, s, num_attr, feature_dim = encoded_data.shape

        # Candidate forces from provider
        F_set = self.force_provider(ais_data)  # [b, s, num_forces, 3]
        num_forces = F_set.shape[2]

        # Flatten batch/time dims for attention
        Q = encoded_data.view(b * s, num_attr, feature_dim)  # [b*s, num_attr, d]
        K = self.force_proj(F_set.view(b * s, num_forces, 3))  # [b*s, num_forces, d]
        V = K  # same as K (forces as both key and value)

        # Cross-attention: AIS queries attend to forces
        attn_output, attn_weights = self.cross_attn(Q, K, V)
        # attn_output: [b*s, num_attr, d]
        # attn_weights: [b*s, num_attr, num_forces]

        # Get attention scores per force
        attn_scores = attn_weights.view(b, s, num_attr, num_forces)

        # Aggregate across AIS attributes to get a single weighted force vector per timestep
        attn_mean = attn_scores.mean(dim=2)  # [b, s, num_forces]
        attn_norm = torch.softmax(attn_mean, dim=-1)  # normalize attention over forces

        # Weighted sum of original forces
        aggregated_forces = torch.sum(F_set * attn_norm.unsqueeze(-1), dim=2)  # [b, s, 3]

        return aggregated_forces, attn_scores
