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


class LossCalculator:

    @staticmethod
    def get_loss_type(loss_type: str,
                      training: bool,
                      full_pos_pred: torch.Tensor, full_pos_true: torch.Tensor,
                      pos_pred: torch.Tensor, pos_true: torch.Tensor,
                      deltas_pred: torch.Tensor, deltas_true: torch.Tensor,
                      total_consistency_loss: torch.Tensor,
                      forces_pred: torch.Tensor, forces_true: torch.Tensor) -> LossOutput:

        pos_distance = LossCalculator._calc_hyp(
            pos_pred[:, :, AISColDict.NORTHERN_DELTA.value],
            pos_true[:, :, AISColDict.NORTHERN_DELTA.value],
            pos_pred[:, :, AISColDict.EASTERN_DELTA.value],
            pos_true[:, :, AISColDict.EASTERN_DELTA.value])
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

        total_loss = pos_distance_loss + 10 * delta_hyp_loss  # + 1000 * total_consistency_loss + 10 * force_loss

        if not training:
            full_pos_distance = LossCalculator._calc_hyp(
                full_pos_pred[:, :, AISColDict.NORTHERN_DELTA.value],
                full_pos_true[:, :, AISColDict.NORTHERN_DELTA.value],
                full_pos_pred[:, :, AISColDict.EASTERN_DELTA.value],
                full_pos_true[:, :, AISColDict.EASTERN_DELTA.value])
            frechet_distance_loss = LossCalculator._frechet_distance(full_pos_distance)
        else:
            frechet_distance_loss = 0.0

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
        )

    @staticmethod
    def _calc_hyp(northern_pred, northern_true, eastern_pred, eastern_true) -> torch.Tensor:
        dx = northern_pred - northern_true
        dy = eastern_pred - eastern_true
        eps = 1e-6
        return torch.sqrt(dx * dx + dy * dy + eps)

    @staticmethod
    def _frechet_distance(dist: torch.Tensor) -> float:
        T = dist.shape[0]
        ca = torch.empty((T, T), device=dist.device)

        ca[0, 0] = dist[0, 0]

        for i in range(1, T):
            ca[i, 0] = torch.max(ca[i - 1, 0], dist[i, 0])

        for j in range(1, T):
            ca[0, j] = torch.max(ca[0, j - 1], dist[0, j])

        for i in range(1, T):
            for j in range(1, T):
                ca[i, j] = torch.max(
                    torch.min(torch.stack([
                        ca[i - 1, j],
                        ca[i - 1, j - 1],
                        ca[i, j - 1],
                    ])),
                    dist[i, j]
                )

        return ca[-1, -1].item()


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
    count: int = 0

    def add_batch(self, loss, batch_size: int):
        self.mae.add(loss.mae, batch_size)
        self.count += batch_size

    def average(self) -> "LossAccumulator":
        return LossAccumulator(
            mae=self.mae.average(self.count),
            count=self.count,
        )

    @staticmethod
    def csv_headers(include_epoch: bool = False):
        headers = []
        if include_epoch:
            headers.append("epoch")
        headers += LossTotals.headers("mae")
        return headers

    def csv_row(self, epoch: int | None = None):
        row = []
        if epoch is not None:
            row.append(epoch)

        row += self.mae.as_list()
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
