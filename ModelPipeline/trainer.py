from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ModelData.i_model_data_upload_handler import IModelDataUploadHandler
from ModelUtils.data_processor import Config as DataProcessorConfig
import os
import datetime as dt

from ModelUtils.loss_calculator import LossAccumulator


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

    def __init__(self, model: nn.Module, train_data_loader: DataLoader, validation_data_loader: DataLoader, test_data_loader: DataLoader, upload_handler: IModelDataUploadHandler, data_processor_cfg: DataProcessorConfig, config: Config):
        self._cfg = config
        self._data_description = f"{data_processor_cfg.masking_strategy}_{data_processor_cfg.masking_percentage}"

        self._run_name = f"run_{model}_{self._data_description}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join(self._cfg.output_dir, "Tensorboard", self._run_name)
        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=1)

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

            if (epoch + 1) % self._cfg.validation_every_n_epochs == 0 and epoch != 0:
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

        with torch.enable_grad():
            for batch_no, batch in enumerate(it, start=1):
                self._optimizer.zero_grad()

                loss, _ = self._model.forward(batch)
                loss.mse.total_loss.backward()
                self._optimizer.step()
                self._global_training_step += 1
                self._writer.add_scalar("train/total", loss.mae.total_loss.item(), self._global_training_step)
                self._writer.add_scalar("train/delta_hyp", loss.mae.delta_hyp_loss.item(), self._global_training_step)
                self._writer.add_scalar("train/pos_distance", loss.mae.pos_distance_loss.item(),
                                        self._global_training_step)
                self._writer.add_scalar("train/consistency", loss.mae.consistency_loss.item(),
                                        self._global_training_step)
                self._writer.add_scalar("train/force", loss.mae.force_loss.item(),
                                        self._global_training_step)

                total_loss += loss.mse.total_loss.item()
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
                loss, _ = self._model.forward(batch)

                loss = loss.mae.total_loss
                total_loss += loss.item()
                average_loss = total_loss / batch_no

                self._writer.add_scalar("validation/total", loss.item(), self._global_validation_step)
                self._writer.add_scalar("validation/avg", average_loss, epoch_no)

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
        acc = LossAccumulator()

        it = tqdm(self._test_data_loader, mininterval=5.0, maxinterval=50.0)

        with torch.no_grad():
            for batch in it:
                loss, _ = self._model.forward(batch)
                batch_size = batch.observed_data.size(0)

                acc.add_batch(loss, batch_size)
                it.set_postfix({"epoch": epoch_no}, refresh=False)

        it.close()
        avg = acc.average()

        for name, val in avg.mae.as_dict().items():
            self._writer.add_scalar(f"test/{name}", val, epoch_no)

        print(
            f"Test Epoch {epoch_no} Complete. "
            f"Avg dist Loss: {avg.mae.pos_dist:.4f}"
        )

        path = f"{self._cfg.output_dir}/Training/{self._run_name}.csv"
        avg.write_csv(path, epoch=epoch_no)
