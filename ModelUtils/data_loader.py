from torch.utils.data import DataLoader
from ModelTypes.ais_dataset_masked import AISDatasetMasked
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
    _data_processor: DataProcessor
    _cfg: Config

    def __init__(self, data_processor: DataProcessor, config: Config):
        self._cfg = config
        self._data_processor = data_processor

    def get_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        dates = [self._cfg.start_date + dt.timedelta(days=i)
                 for i in range(0, (self._cfg.end_date - self._cfg.start_date).days + 1, self._cfg.date_step)]

        dataset = self._data_processor.get_masked_data(dates)

        train_data, test_data = AisDataLoader.split_dataset(
            self._cfg.train_split, dataset)

        train_loader = DataLoader(train_data, batch_size=self._cfg.batch_size,
                                  shuffle=self._cfg.shuffle,
                                  num_workers=self._cfg.num_workers)
        test_loader = DataLoader(test_data, batch_size=self._cfg.batch_size,
                                 shuffle=self._cfg.shuffle,
                                 num_workers=self._cfg.num_workers)
        return train_loader, test_loader

    @staticmethod
    def split_dataset(train_split: float, dataset: AISDatasetMasked) -> tuple[AISDatasetMasked, AISDatasetMasked]:
        indices = np.arange(len(dataset))
        np.random.default_rng(seed=42).shuffle(indices)
        train_size = int(len(indices) * train_split)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        def extract_subset(idxs):
            return AISDatasetMasked(
                dataset.data[idxs],
                dataset.labels[idxs],
                dataset.masks[idxs],
                dataset.num_masked_values,
                dataset.num_values_in_sequence,)

        return extract_subset(train_indices), extract_subset(test_indices)
