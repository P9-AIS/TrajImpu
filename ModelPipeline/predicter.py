import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import csv
import datetime as dt
from dataclasses import dataclass
from ModelData.i_model_data_upload_handler import IModelDataUploadHandler
from ModelUtils.loss_calculator import LossAccumulator
from ModelUtils.data_processor import Config as DataProcessorConfig


@dataclass
class Config:
    output_dir: str = "Outputs"


class Predicter:
    def __init__(
        self,
        model: torch.nn.Module,
        upload_handler: IModelDataUploadHandler,
        dataloader: DataLoader,
        data_processor_cfg: DataProcessorConfig,
        cfg: Config,
    ):
        self._model = model
        self._upload_handler = upload_handler
        self._dataloader = dataloader
        self._cfg = cfg
        self._data_description = f"{data_processor_cfg.masking_strategy}_{data_processor_cfg.masking_percentage}"
        self._run_name = f"predict_{str(model)}_{self._data_description}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def predict(self):
        model_name = f"{str(self._model)}_{self._data_description}"
        self._model.eval()
        self._upload_handler.reset_predictions(model_name)

        acc = LossAccumulator()

        all_pred_northerns = []
        all_pred_easterns = []
        all_true_northerns = []
        all_true_easterns = []
        all_masks = []

        it = tqdm(self._dataloader, mininterval=2.0)

        with torch.no_grad():
            for batch in it:
                loss, _, (pred_lats, pred_lons, true_lats, true_lons) = self._model.forward(batch)
                batch_size = batch.observed_data.size(0)

                acc.add_batch(loss, batch_size)

                all_pred_northerns.append(pred_lats.cpu())
                all_pred_easterns.append(pred_lons.cpu())
                all_true_northerns.append(true_lats.cpu())
                all_true_easterns.append(true_lons.cpu())
                all_masks.append(batch.masks.cpu())

        # concat
        all_pred_northerns = torch.cat(all_pred_northerns, dim=0)
        all_pred_easterns = torch.cat(all_pred_easterns, dim=0)
        all_true_northerns = torch.cat(all_true_northerns, dim=0)
        all_true_easterns = torch.cat(all_true_easterns, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        # upload
        self._upload_handler.upload_predictions(
            model_name=model_name,
            masks=all_masks,
            predicted_northerns=all_pred_northerns,
            predicted_easterns=all_pred_easterns,
            true_northerns=all_true_northerns,
            true_easterns=all_true_easterns,
        )

        avg = acc.average()

        path = os.path.join(self._cfg.output_dir, "Predictions", f"{self._run_name}.csv")
        avg.write_csv(path)

        print(f"Predictions saved to CSV: {path}")
        print(f"Average MAE pos_dist loss: {avg.mae.pos_dist:.4f}")

        return (
            all_pred_northerns,
            all_pred_easterns,
            all_true_northerns,
            all_true_easterns,
            avg.mae.as_dict(),
        )
