import torch
from dataclasses import dataclass
from Model.ais_decoder import ExtraDecodeOutput
from ModelTypes.ais_col_dict import AISColDict
from ModelUtils.geo_utils import GeoUtils
import torch.nn.functional as F


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    lat_loss: torch.Tensor
    lon_loss: torch.Tensor
    cog_loss: torch.Tensor
    heading_loss: torch.Tensor
    draught_loss: torch.Tensor
    sog_loss: torch.Tensor
    rot_loss: torch.Tensor
    vessel_type_loss: torch.Tensor

    def __str__(self) -> str:
        return (f"Total Loss: {self.total_loss.item():.4f}\n"
                f"Lat Loss: {self.lat_loss.item():.4f}\n"
                f"Lon Loss: {self.lon_loss.item():.4f}\n"
                f"COG Loss: {self.cog_loss.item():.4f}\n"
                f"Heading Loss: {self.heading_loss.item():.4f}\n"
                f"Draught Loss: {self.draught_loss.item():.4f}\n"
                f"SOG Loss: {self.sog_loss.item():.4f}\n"
                f"ROT Loss: {self.rot_loss.item():.4f}\n"
                f"Vessel Type Loss: {self.vessel_type_loss.item():.4f}")


@dataclass
class LossTypes:
    mse: LossOutput
    mae: LossOutput
    # smape: LossOutput


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

    @staticmethod
    def get_loss(loss_func: str, prediction, truth, prediction_extra=None) -> torch.Tensor:
        if loss_func == "mse":
            return torch.nn.functional.mse_loss(prediction, truth)
        elif loss_func == "mae":
            return torch.nn.functional.l1_loss(prediction, truth)
        # elif loss_func == "smape":
        #     # Symmetric Mean Absolute Percentage Error
        #     class SMAPELoss(torch.nn.Module):
        #         def __init__(self):
        #             super(SMAPELoss, self).__init__()

        #         def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        #             numerator = torch.abs(y_pred - y_true)
        #             denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2.0
        #             smape = numerator / (denominator + 1e-8)  # avoid division by zero
        #             return torch.mean(smape)
        #     return SMAPELoss()
        elif loss_func == "cross_entropy" and prediction_extra is not None:
            y_pred = torch.stack([extra.vessel_type_prob_logits for extra in prediction_extra], dim=0)
            y_pred = y_pred.view(-1, y_pred.shape[-1])  # flatten to [b*s, num_classes]
            y_true = truth.long().squeeze(-1).view(-1)  # [b*s]
            return F.cross_entropy(y_pred, y_true)
        elif loss_func == "cyclical":
            diff = torch.abs(prediction - truth)
            cyclical_diff = torch.min(diff, 360 - diff)
            return torch.mean(cyclical_diff)
        else:
            raise ValueError(f"Unsupported loss function: {loss_func}")

    @staticmethod
    def get_loss_type(loss_type: str, prediction, truth, prediction_extra) -> LossOutput:

        lat_loss = LossCalculator.get_loss(
            loss_type,
            prediction[:, :, AISColDict.LATITUDE.value:AISColDict.LATITUDE.value+1],
            truth[:, :, AISColDict.LATITUDE.value:AISColDict.LATITUDE.value+1])

        lon_loss = LossCalculator.get_loss(
            loss_type,
            prediction[:, :, AISColDict.LONGITUDE.value:AISColDict.LONGITUDE.value+1],
            truth[:, :, AISColDict.LONGITUDE.value:AISColDict.LONGITUDE.value+1])

        cog_loss = LossCalculator.get_loss(
            "cyclical",
            prediction[:, :, AISColDict.COG.value:AISColDict.COG.value+1],
            truth[:, :, AISColDict.COG.value:AISColDict.COG.value+1])

        heading_loss = LossCalculator.get_loss(
            "cyclical",
            prediction[:, :, AISColDict.HEADING.value:AISColDict.HEADING.value+1],
            truth[:, :, AISColDict.HEADING.value:AISColDict.HEADING.value+1])

        draught_loss = LossCalculator.get_loss(
            loss_type,
            prediction[:, :, AISColDict.DRAUGHT.value:AISColDict.DRAUGHT.value+1],
            truth[:, :, AISColDict.DRAUGHT.value:AISColDict.DRAUGHT.value+1])

        sog_loss = LossCalculator.get_loss(
            loss_type,
            prediction[:, :, AISColDict.SOG.value:AISColDict.SOG.value+1],
            truth[:, :, AISColDict.SOG.value:AISColDict.SOG.value+1])

        rot_loss = LossCalculator.get_loss(
            loss_type,
            prediction[:, :, AISColDict.ROT.value:AISColDict.ROT.value+1],
            truth[:, :, AISColDict.ROT.value:AISColDict.ROT.value+1])

        vessel_type_loss = LossCalculator.get_loss(
            "cross_entropy",
            prediction[:, :, AISColDict.VESSEL_TYPE.value:AISColDict.VESSEL_TYPE.value+1],
            truth[:, :, AISColDict.VESSEL_TYPE.value:AISColDict.VESSEL_TYPE.value+1],
            prediction_extra)

        total_loss = (lat_loss + lon_loss + cog_loss + heading_loss +
                      draught_loss + sog_loss + rot_loss + vessel_type_loss)

        return LossOutput(
            total_loss=total_loss,
            lat_loss=lat_loss,
            lon_loss=lon_loss,
            cog_loss=cog_loss,
            heading_loss=heading_loss,
            draught_loss=draught_loss,
            sog_loss=sog_loss,
            rot_loss=rot_loss,
            vessel_type_loss=vessel_type_loss
        )

    def calculate_loss(self, imputed: torch.Tensor, imputed_extra: list[ExtraDecodeOutput], ground_truth: torch.Tensor) -> LossTypes:
        return LossTypes(
            mse=self.get_loss_type("mse", imputed, ground_truth, imputed_extra),
            mae=self.get_loss_type("mae", imputed, ground_truth, imputed_extra),
            # smape=self.get_loss_type("smape", imputed, ground_truth)
        )

    def _calc_spatial_loss(self, imputed_lat, ground_truth_lat, imputed_lon, ground_truth_lon) -> torch.Tensor:
        dist = torch.hypot(imputed_lat - ground_truth_lat, imputed_lon - ground_truth_lon)
        return torch.mean(dist)
