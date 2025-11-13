import torch
from dataclasses import dataclass
from ModelUtils.geo_utils import GeoUtils


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    spatial_loss: torch.Tensor
    cog_loss: torch.Tensor
    heading_loss: torch.Tensor
    draught_loss: torch.Tensor
    sog_loss: torch.Tensor
    rot_loss: torch.Tensor
    vessel_type_loss: torch.Tensor


@dataclass
class Config:
    spatial_weight: float = 1.0
    cog_weight: float = 1.0
    heading_weight: float = 1.0
    draught_weight: float = 1.0
    sog_weight: float = 1.0
    rot_weight: float = 1.0
    vessel_type_weight: float = 1.0


class LossCalculator:
    _config: Config

    def __init__(self, config: Config):
        self._config = config

    def calculate_loss(self, imputed: torch.Tensor, ground_truth: torch.Tensor) -> LossOutput:
        spatial_loss = self.calc_spatial_loss(
            imputed[:, :, :2],
            ground_truth[:, :, :2]
        ) * self._config.spatial_weight
        cog_loss = self.calc_cog_loss(
            imputed[:, :, 2:3],
            ground_truth[:, :, 2:3]
        ) * self._config.cog_weight
        heading_loss = self.calc_heading_loss(
            imputed[:, :, 3:4],
            ground_truth[:, :, 3:4]
        ) * self._config.heading_weight
        draught_loss = self.calc_draught_loss(
            imputed[:, :, 4:5],
            ground_truth[:, :, 4:5]
        ) * self._config.draught_weight
        sog_loss = self.calc_sog_loss(
            imputed[:, :, 5:6],
            ground_truth[:, :, 5:6]
        ) * self._config.sog_weight
        rot_loss = self.calc_rot_loss(
            imputed[:, :, 6:7],
            ground_truth[:, :, 6:7]
        ) * self._config.rot_weight
        vessel_type_loss = self.calc_vessel_type_loss(
            imputed[:, :, 7:],
            ground_truth[:, :, 7:].long().squeeze(-1)
        ) * self._config.vessel_type_weight

        total_loss = (spatial_loss + cog_loss + heading_loss + draught_loss + sog_loss + rot_loss + vessel_type_loss)

        return LossOutput(
            total_loss=total_loss,
            spatial_loss=spatial_loss,
            cog_loss=cog_loss,
            heading_loss=heading_loss,
            draught_loss=draught_loss,
            sog_loss=sog_loss,
            rot_loss=rot_loss,
            vessel_type_loss=vessel_type_loss
        )

    def calc_spatial_loss(self, imputed_spatial: torch.Tensor, ground_truth_spatial: torch.Tensor) -> torch.Tensor:
        distances = GeoUtils.haversine_distances_m(
            pos1=imputed_spatial,
            pos2=ground_truth_spatial
        )
        return torch.mean(distances)

    def calc_cog_loss(self, imputed_cog: torch.Tensor, ground_truth_cog: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(imputed_cog, ground_truth_cog)

    def calc_heading_loss(self, imputed_heading: torch.Tensor, ground_truth_heading: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(imputed_heading, ground_truth_heading)

    def calc_draught_loss(self, imputed_draught: torch.Tensor, ground_truth_draught: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(imputed_draught, ground_truth_draught)

    def calc_sog_loss(self, imputed_sog: torch.Tensor, ground_truth_sog: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(imputed_sog, ground_truth_sog)

    def calc_rot_loss(self, imputed_rot: torch.Tensor, ground_truth_rot: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(imputed_rot, ground_truth_rot)

    def calc_vessel_type_loss(self, imputed_vessel_type: torch.Tensor, ground_truth_vessel_type: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            imputed_vessel_type,
            ground_truth_vessel_type.long().squeeze(-1)
        )
