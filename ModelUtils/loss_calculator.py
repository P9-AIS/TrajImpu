import torch
from dataclasses import dataclass
from Model.ais_decoder import ExtraDecodeOutput
from ModelTypes.ais_col_dict import AISColDict
from ModelUtils.geo_utils import GeoUtils
import torch.nn.functional as F


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
    _extra_decode_output: ExtraDecodeOutput

    def __init__(self, config: Config):
        self._config = config

    def calculate_loss(self, imputed: torch.Tensor, imputed_extra: list[ExtraDecodeOutput], ground_truth: torch.Tensor) -> LossOutput:
        spatial_loss = self._calc_spatial_loss(
            imputed[:, :, :AISColDict.LONGITUDE.value+AISColDict.LATITUDE.value+1],
            ground_truth[:, :, :AISColDict.LONGITUDE.value+AISColDict.LATITUDE.value+1]
        ) * self._config.spatial_weight
        cog_loss = self._calc_cog_loss(
            imputed[:, :, AISColDict.COG.value:AISColDict.COG.value+1],
            ground_truth[:, :, AISColDict.COG.value:AISColDict.COG.value+1]
        ) * self._config.cog_weight
        heading_loss = self._calc_heading_loss(
            imputed[:, :, AISColDict.HEADING.value:AISColDict.HEADING.value+1],
            ground_truth[:, :, AISColDict.HEADING.value:AISColDict.HEADING.value+1]
        ) * self._config.heading_weight
        draught_loss = self._calc_draught_loss(
            imputed[:, :, AISColDict.DRAUGHT.value:AISColDict.DRAUGHT.value+1],
            ground_truth[:, :, AISColDict.DRAUGHT.value:AISColDict.DRAUGHT.value+1]
        ) * self._config.draught_weight
        sog_loss = self._calc_sog_loss(
            imputed[:, :, AISColDict.SOG.value:AISColDict.SOG.value+1],
            ground_truth[:, :, AISColDict.SOG.value:AISColDict.SOG.value+1]
        ) * self._config.sog_weight
        rot_loss = self._calc_rot_loss(
            imputed[:, :, AISColDict.ROT.value:AISColDict.ROT.value+1],
            ground_truth[:, :, AISColDict.ROT.value:AISColDict.ROT.value+1],
        ) * self._config.rot_weight
        vessel_type_loss = self._calc_vessel_type_loss(
            imputed_extra,
            ground_truth[:, :, AISColDict.VESSEL_TYPE.value:AISColDict.VESSEL_TYPE.value+1]
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

    def _calc_spatial_loss(self, imputed_spatial: torch.Tensor, ground_truth_spatial: torch.Tensor) -> torch.Tensor:
        distances = GeoUtils.haversine_distances_m(
            pos1=imputed_spatial,
            pos2=ground_truth_spatial
        )
        return torch.mean(distances)

    def _calc_cog_loss(self, imputed_cog: torch.Tensor, ground_truth_cog: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(imputed_cog, ground_truth_cog)

    def _calc_heading_loss(self, imputed_heading: torch.Tensor, ground_truth_heading: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(imputed_heading, ground_truth_heading)

    def _calc_draught_loss(self, imputed_draught: torch.Tensor, ground_truth_draught: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(imputed_draught, ground_truth_draught)

    def _calc_sog_loss(self, imputed_sog: torch.Tensor, ground_truth_sog: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(imputed_sog, ground_truth_sog)

    def _calc_rot_loss(self, imputed_rot: torch.Tensor, ground_truth_rot: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(imputed_rot, ground_truth_rot)

    def _calc_vessel_type_loss(self, imputed_extra: list[ExtraDecodeOutput], ground_truth_vessel_type: torch.Tensor) -> torch.Tensor:
        y_pred = torch.stack([extra.vessel_type_prob_logits for extra in imputed_extra], dim=0)
        # If imputed_extra is length batch*seq already, then torch.cat is fine
        # Otherwise stack along batch dimension
        y_pred = y_pred.view(-1, y_pred.shape[-1])  # flatten to [b*s, num_classes]

        # Flatten ground truth
        y_true = ground_truth_vessel_type.long().squeeze(-1).view(-1)  # [b*s]

        # Optional: shift indices if your labels start at non-zero
        # min_val = 6
        # y_true = y_true - min_val

        return F.cross_entropy(y_pred, y_true)
