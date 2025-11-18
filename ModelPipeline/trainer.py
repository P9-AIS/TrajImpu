from dataclasses import dataclass

import torch
from tqdm import tqdm
from Model.model import Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Config:
    output_dir: str = "Outputs"
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_patience: int = 3
    validation_every_n_epochs: int = 5


class Trainer:
    _cfg: Config
    _train_data_loader: DataLoader
    _validation_data_loader: DataLoader
    _optimizer: torch.optim.Optimizer

    def __init__(self, model: Model, train_data_loader: DataLoader, validation_data_loader: DataLoader, config: Config):
        self._writer = SummaryWriter(flush_secs=1)
        self._cfg = config
        self._train_data_loader = train_data_loader
        self._validation_data_loader = validation_data_loader
        self._model = model
        self._optimizer = torch.optim.AdamW(self._model.parameters(),
                                            lr=self._cfg.learning_rate,
                                            weight_decay=self._cfg.weight_decay)
        self._global_training_step = 0
        self._global_validation_step = 0

    def train(self):
        print("Training the model...")

        best_train_loss = float('inf')
        best_average_validation_loss = float('inf')
        epochs_since_improvement = 0

        for epoch in range(self._cfg.num_epochs):
            print(f"Epoch {epoch + 1}/{self._cfg.num_epochs}")
            average_train_loss = self._run_training_batches(epoch)

            if average_train_loss < best_train_loss:
                best_train_loss = average_train_loss

            if epoch % self._cfg.validation_every_n_epochs == 0:
                average_validation_loss = self._run_validation_batches(epoch)

                if average_validation_loss < best_average_validation_loss:
                    best_average_validation_loss = average_validation_loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

            if epochs_since_improvement >= self._cfg.validation_patience:
                print("Early stopping triggered.")
                break

    def _run_training_batches(self, epoch_no: int) -> float:
        it = tqdm(self._train_data_loader, mininterval=5.0, maxinterval=50.0)

        total_loss = 0.0
        average_loss = 0.0
        self._model.train()

        with torch.enable_grad():
            for batch_no, batch in enumerate(it, start=1):
                self._optimizer.zero_grad()
                loss = self._model.forward(batch)
                loss.total_loss.backward()
                self._optimizer.step()
                self._global_training_step += 1
                self._writer.add_scalar("loss/train", loss.total_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_avg", average_loss, epoch_no)
                self._writer.add_scalar("loss/train_lat", loss.lat_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_lon", loss.lon_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_cog", loss.cog_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_sog", loss.sog_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_rot", loss.rot_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_heading", loss.heading_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_draught", loss.draught_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_vessel_type", loss.vessel_type_loss.item(),
                                        self._global_training_step)
                self._writer.add_scalar("loss/train_haversine", loss.haversine_loss.item(), self._global_training_step)

                total_loss += loss.total_loss.item()
                average_loss = total_loss / batch_no

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": average_loss,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

        it.close()
        return average_loss

    def _run_validation_batches(self, epoch_no: int) -> float:
        it = tqdm(self._validation_data_loader, mininterval=5.0, maxinterval=50.0)

        total_loss = 0.0
        average_loss = 0.0
        self._model.eval()

        with torch.no_grad():
            for batch_no, batch in enumerate(it, start=1):
                loss = self._model.forward(batch).total_loss

                total_loss += loss.item()
                average_loss = total_loss / batch_no

                self._writer.add_scalar("loss/validation", loss.item(), self._global_validation_step)
                self._writer.add_scalar("loss/validation_avg", average_loss, epoch_no)

                self._global_validation_step += 1

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": average_loss,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

        it.close()
        return average_loss
