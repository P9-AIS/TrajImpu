import csv
import os
import torch
from dataclasses import dataclass, field
from ModelTypes.ais_col_dict import AISColDict


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    pos_distance_loss: torch.Tensor
    delta_hyp_loss: torch.Tensor
    consistency_loss: torch.Tensor
    force_loss: torch.Tensor
    frechet_distance_loss: float

    def __str__(self) -> str:
        return (f"Total Loss: {self.total_loss.item():.4f}\n"
                f"Position Distance Loss: {self.pos_distance_loss.item():.4f}\n"
                f"Delta Hyp Loss: {self.delta_hyp_loss.item():.4f}\n"
                f"Consistency Loss: {self.consistency_loss.item():.4f}\n"
                f"Force Loss: {self.force_loss.item():.4f}\n"
                f"Frechet Distance Loss: {self.frechet_distance_loss:.4f}")


@dataclass
class LossTypes:
    mse: LossOutput
    mae: LossOutput
    smape: LossOutput


class LossCalculator:

    @staticmethod
    def get_loss_type(loss_type: str,
                      training: bool,
                      full_pos_pred: torch.Tensor, full_pos_true: torch.Tensor,
                      pos_pred: torch.Tensor, pos_true: torch.Tensor,
                      deltas_pred: torch.Tensor, deltas_true: torch.Tensor,
                      total_consistency_loss: torch.Tensor,
                      forces_pred: torch.Tensor, forces_true: torch.Tensor) -> LossOutput:

        pos_distance = LossCalculator._euclidean_distance(pos_pred, pos_true)
        pos_distance_zero = torch.zeros_like(pos_distance)

        pos_distance_loss = LossCalculator.get_loss(
            loss_type,
            pos_distance,
            pos_distance_zero)

        delta_hyp = LossCalculator._calc_hyp(
            deltas_pred[:, :, AISColDict.NORTHERN_DELTA.value],
            deltas_true[:, :, AISColDict.NORTHERN_DELTA.value],
            deltas_pred[:, :, AISColDict.EASTERN_DELTA.value],
            deltas_true[:, :, AISColDict.EASTERN_DELTA.value])
        delta_hyp_zero = torch.zeros_like(delta_hyp)

        delta_hyp_loss = LossCalculator.get_loss(
            loss_type,
            delta_hyp,
            delta_hyp_zero)

        force_loss = LossCalculator.get_loss(loss_type, forces_pred, forces_true)

        total_loss = pos_distance_loss + delta_hyp_loss + 1000 * total_consistency_loss + 10 * force_loss

        if not training:
            frechet_distance_loss = LossCalculator._frechet_distance(full_pos_pred, full_pos_true)
        else:
            frechet_distance_loss = torch.tensor(0.0, device=full_pos_pred.device)

        return LossOutput(
            total_loss=total_loss,
            pos_distance_loss=pos_distance_loss,
            delta_hyp_loss=delta_hyp_loss,
            consistency_loss=total_consistency_loss,
            force_loss=force_loss,
            frechet_distance_loss=frechet_distance_loss
        )

    @staticmethod
    def get_loss(loss_func: str, prediction, truth) -> torch.Tensor:
        if loss_func == "mse":
            return torch.nn.functional.mse_loss(prediction, truth)
        elif loss_func == "mae":
            return torch.nn.functional.l1_loss(prediction, truth)
        elif loss_func == "smape":
            # SMAPE formula: 2 * |y_pred - y_true| / (|y_pred| + |y_true|)
            numerator = torch.abs(prediction - truth)
            denominator = torch.abs(prediction) + torch.abs(truth)

            epsilon = 1e-8
            return 2 * torch.mean(numerator / (denominator + epsilon))
        else:
            raise ValueError(f"Unsupported loss function: {loss_func}")

    def calculate_loss(self,
                       training: bool,
                       full_pos_pred: torch.Tensor, full_pos_true: torch.Tensor,
                       pos_pred: torch.Tensor, pos_true: torch.Tensor,
                       deltas_pred: torch.Tensor, deltas_true: torch.Tensor,
                       total_consistency_loss: torch.Tensor,
                       decoded_forces: torch.Tensor, true_forces: torch.Tensor) -> LossTypes:
        return LossTypes(
            mse=self.get_loss_type("mse", training, full_pos_pred, full_pos_true, pos_pred, pos_true, deltas_pred, deltas_true,
                                   total_consistency_loss, decoded_forces, true_forces),
            mae=self.get_loss_type("mae", training, full_pos_pred, full_pos_true, pos_pred, pos_true, deltas_pred, deltas_true,
                                   total_consistency_loss, decoded_forces, true_forces),
            smape=self.get_loss_type("smape", training, full_pos_pred, full_pos_true, pos_pred, pos_true, deltas_pred, deltas_true,
                                     total_consistency_loss, decoded_forces, true_forces)
        )

    @staticmethod
    def _calc_hyp(imputed_lat, ground_truth_lat, imputed_lon, ground_truth_lon) -> torch.Tensor:
        return torch.hypot(imputed_lat - ground_truth_lat, imputed_lon - ground_truth_lon)

    @staticmethod
    def _euclidean_distance(pos_pred, pos_true) -> torch.Tensor:
        lat1 = pos_pred[..., 0]
        lon1 = pos_pred[..., 1]
        lat2 = pos_true[..., 0]
        lon2 = pos_true[..., 1]

        R = 6371000.0  # Earth radius in meters
        lat1 = torch.deg2rad(lat1)
        lon1 = torch.deg2rad(lon1)
        lat2 = torch.deg2rad(lat2)
        lon2 = torch.deg2rad(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        mean_lat = (lat1 + lat2) / 2.0

        x = dlon * torch.cos(mean_lat)
        y = dlat

        return R * torch.sqrt(x * x + y * y)

    @staticmethod
    def _frechet_distance(pred: torch.Tensor, true: torch.Tensor) -> float:
        dist = LossCalculator._euclidean_distance(pred, true)  # (T, T)
        T = dist.shape[0]

        ca = torch.full((T, T), -1.0, device=pred.device)

        def recurse(i, j):
            if ca[i, j] > -0.5:
                return ca[i, j]
            elif i == 0 and j == 0:
                ca[i, j] = dist[0, 0]
            elif i > 0 and j == 0:
                ca[i, j] = torch.max(recurse(i - 1, 0), dist[i, 0])
            elif i == 0 and j > 0:
                ca[i, j] = torch.max(recurse(0, j - 1), dist[0, j])
            elif i > 0 and j > 0:
                ca[i, j] = torch.max(
                    torch.min(torch.stack([
                        recurse(i - 1, j),
                        recurse(i - 1, j - 1),
                        recurse(i, j - 1)
                    ])),
                    dist[i, j]
                )
            else:
                ca[i, j] = float("inf")
            return ca[i, j]

        return recurse(T - 1, T - 1).item()


@dataclass
class LossTotals:
    pos_dist: float = 0.0
    delta_hyp: float = 0.0
    frechet: float = 0.0

    def add(self, loss, batch_size: int):
        self.pos_dist += loss.pos_distance_loss.item() * batch_size
        self.delta_hyp += loss.delta_hyp_loss.item() * batch_size
        self.frechet += loss.frechet_distance_loss * batch_size

    def average(self, count: int) -> "LossTotals":
        if count == 0:
            return self
        return LossTotals(
            pos_dist=self.pos_dist / count,
            delta_hyp=self.delta_hyp / count,
            frechet=self.frechet / count,
        )

    def as_dict(self):
        return {
            "pos_dist": self.pos_dist,
            "delta_hyp": self.delta_hyp,
            "frechet": self.frechet,
        }

    def as_list(self):
        return [self.pos_dist, self.delta_hyp, self.frechet]

    @staticmethod
    def headers(prefix: str):
        return [
            f"{prefix}_pos_dist",
            f"{prefix}_delta_hyp",
            f"{prefix}_frechet",
        ]


@dataclass
class LossAccumulator:
    mae: LossTotals = field(default_factory=LossTotals)
    smape: LossTotals = field(default_factory=LossTotals)
    count: int = 0

    def add_batch(self, loss, batch_size: int):
        self.mae.add(loss.mae, batch_size)
        self.smape.add(loss.smape, batch_size)
        self.count += batch_size

    def average(self) -> "LossAccumulator":
        return LossAccumulator(
            mae=self.mae.average(self.count),
            smape=self.smape.average(self.count),
            count=self.count,
        )

    @staticmethod
    def csv_headers(include_epoch: bool = False):
        headers = []
        if include_epoch:
            headers.append("epoch")
        headers += LossTotals.headers("mae")
        headers += LossTotals.headers("smape")
        return headers

    def csv_row(self, epoch: int | None = None):
        row = []
        if epoch is not None:
            row.append(epoch)

        row += self.mae.as_list()
        row += self.smape.as_list()
        return [f"{v:.6f}" if isinstance(v, float) else v for v in row]

    def write_csv(
        self,
        path: str,
        epoch: int | None = None,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_exists = os.path.isfile(path)

        with open(path, mode="a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(self.csv_headers(include_epoch=epoch is not None))

            writer.writerow(self.csv_row(epoch))
