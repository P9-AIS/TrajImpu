from torch.utils.data import DataLoader
from ModelTypes.ais_dataset import AISDatasetProcessed
from ModelUtils.data_processor import DataProcessor
from dataclasses import dataclass
import datetime as dt
import numpy as np


@dataclass
class Config:
    batch_size: int
    shuffle: bool
    num_workers: int
    train_split: float
    start_date: dt.date
    end_date: dt.date
    date_step: int


class AisDataLoader:
    @staticmethod
    def get_data_loaders(cfg: Config, data_processor: DataProcessor) -> tuple[DataLoader, DataLoader]:
        dates = [cfg.start_date + dt.timedelta(days=i)
                 for i in range(0, (cfg.end_date - cfg.start_date).days + 1, cfg.date_step)]

        dataset = data_processor.get_processed_data(dates)

        train_data, test_data = AisDataLoader.split_dataset(
            cfg.train_split, dataset)

        train_loader = DataLoader(train_data, batch_size=cfg.batch_size,
                                  shuffle=cfg.shuffle,
                                  num_workers=cfg.num_workers)
        test_loader = DataLoader(test_data, batch_size=cfg.batch_size,
                                 shuffle=cfg.shuffle,
                                 num_workers=cfg.num_workers)
        return train_loader, test_loader

    @staticmethod
    def split_dataset(train_split: float, dataset: AISDatasetProcessed) -> tuple[AISDatasetProcessed, AISDatasetProcessed]:
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        train_size = int(len(indices) * train_split)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        def extract_subset(idxs):
            return AISDatasetProcessed(
                dataset.data[idxs],
                dataset.labels[idxs],
                dataset.masks[idxs],
                dataset.padding_masks[idxs],
            )

        return extract_subset(train_indices), extract_subset(test_indices)
