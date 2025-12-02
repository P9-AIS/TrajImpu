from dataclasses import dataclass

import torch
from tqdm import tqdm
from Model.model import Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ModelData.i_model_data_upload_handler import IModelDataUploadHandler
import os
import csv
import datetime as dt


@dataclass
class Config:
    output_dir: str = "Outputs"
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_patience: int = 3
    validation_every_n_epochs: int = 5
    upload_every_n_steps: int = 100


class Trainer:
    _cfg: Config
    _train_data_loader: DataLoader
    _validation_data_loader: DataLoader
    _optimizer: torch.optim.Optimizer
    _upload_handler: IModelDataUploadHandler

    def __init__(self, model: Model, train_data_loader: DataLoader, validation_data_loader: DataLoader, test_data_loader: DataLoader, upload_handler: IModelDataUploadHandler, config: Config):
        self._writer = SummaryWriter(flush_secs=1)
        self._cfg = config
        self._train_data_loader = train_data_loader
        self._validation_data_loader = validation_data_loader
        self._test_data_loader = test_data_loader
        self._model = model
        self._upload_handler = upload_handler
        self._optimizer = torch.optim.AdamW(self._model.parameters(),
                                            lr=self._cfg.learning_rate,
                                            weight_decay=self._cfg.weight_decay)
        self._global_training_step = 0
        self._global_validation_step = 0
        self._global_test_step = 0
        self._global_upload_step = 0

        self._run_name = f"run_{model}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

            if epoch % self._cfg.validation_every_n_epochs == 0 and epoch != 0:
                average_validation_loss = self._run_validation_batches(epoch)

                self._run_test_batches(epoch)

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

        self._upload_handler.reset_predictions()

        with torch.enable_grad():
            for batch_no, batch in enumerate(it, start=1):
                self._optimizer.zero_grad()

                loss, (pred_lats, pred_lons, true_lats, true_lons) = self._model.forward(batch)
                loss.mse.total_loss.backward()
                self._optimizer.step()
                self._global_training_step += 1
                self._writer.add_scalar("loss/train", loss.mae.total_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_avg_mse", average_loss, epoch_no)
                self._writer.add_scalar("loss/train_lat", loss.mae.lat_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_lon", loss.mae.lon_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_cog", loss.mae.cog_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_sog", loss.mae.sog_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_rot", loss.mae.rot_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_heading", loss.mae.heading_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_draught", loss.mae.draught_loss.item(), self._global_training_step)
                self._writer.add_scalar("loss/train_vessel_type", loss.mae.vessel_type_loss.item(),
                                        self._global_training_step)
                self._writer.add_scalar("loss/train_haversine", loss.mae.haversine_loss.item(),
                                        self._global_training_step)

                total_loss += loss.mse.total_loss.item()
                average_loss = total_loss / batch_no

                if self._global_training_step % self._cfg.upload_every_n_steps == 0:
                    self._global_upload_step += 1
                    self._upload_handler.upload_predictions(
                        step=self._global_upload_step,
                        predicted_lats=pred_lats,
                        predicted_lons=pred_lons,
                        true_lats=true_lats,
                        true_lons=true_lons,
                    )

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
                loss, _ = self._model.forward(batch)

                loss = loss.mae.total_loss
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

    def _run_test_batches(self, epoch_no: int):
        self._model.eval()

        total_losses = {
            "lat": 0.0, "lon": 0.0, "cog": 0.0, "sog": 0.0,
            "rot": 0.0, "heading": 0.0, "draught": 0.0, "vessel_type": 0.0
        }
        count = 0

        it = tqdm(self._test_data_loader, mininterval=5.0, maxinterval=50.0)

        with torch.no_grad():
            for batch in it:
                loss, _ = self._model.forward(batch)
                # Ensure batch[0] exists and has a size attribute
                batch_size = batch[0].size(0)

                total_losses["lat"] += loss.mae.lat_loss.item() * batch_size
                total_losses["lon"] += loss.mae.lon_loss.item() * batch_size
                total_losses["cog"] += loss.mae.cog_loss.item() * batch_size
                total_losses["sog"] += loss.mae.sog_loss.item() * batch_size
                total_losses["rot"] += loss.mae.rot_loss.item() * batch_size
                total_losses["heading"] += loss.mae.heading_loss.item() * batch_size
                total_losses["draught"] += loss.mae.draught_loss.item() * batch_size
                total_losses["vessel_type"] += loss.mae.vessel_type_loss.item() * batch_size

                count += batch_size

                it.set_postfix({"epoch": epoch_no}, refresh=False)

        it.close()

        if count > 0:
            # 1. Calculate averages into a neat dictionary
            avg_losses = {name: val / count for name, val in total_losses.items()}

            # 2. Log to TensorBoard
            for name, avg_loss in avg_losses.items():
                self._writer.add_scalar(f"test/{name}", avg_loss, epoch_no)

            # 3. Print to Console
            print(f"Test Epoch {epoch_no} Complete. Avg Lat Loss: {avg_losses['lat']:.4f}")

            # 4. Write to File (CSV format is best for analysis later)
            dirpath = f"{self._cfg.output_dir}/test_logs"
            os.makedirs(dirpath, exist_ok=True)
            log_path = f"{dirpath}/{self._run_name}.csv"
            file_exists = os.path.isfile(log_path)

            with open(log_path, mode='a', newline='') as f:
                writer = csv.writer(f)

                # If file is new, write the header row first
                if not file_exists:
                    headers = ["epoch"] + list(avg_losses.keys())
                    writer.writerow(headers)

                # Write the data row
                row_data = [epoch_no] + [f"{val:.6f}" for val in avg_losses.values()]
                writer.writerow(row_data)
