from torch.utils.data import DataLoader
from ModelTypes.ais_dataset_masked import AISDatasetMasked
from ModelTypes.ais_stats import AISStats
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
    validation_split: float
    test_split: float
    start_date: dt.date
    end_date: dt.date
    date_step: int


class AisDataLoader:
    _data_processor: DataProcessor
    _cfg: Config

    def __init__(self, data_processor: DataProcessor, config: Config):
        self._cfg = config
        self._data_processor = data_processor

    def get_data_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader, AISStats]:
        dates = [self._cfg.start_date + dt.timedelta(days=i)
                 for i in range(0, (self._cfg.end_date - self._cfg.start_date).days + 1, self._cfg.date_step)]

        dataset = self._data_processor.get_masked_data(dates)

        train_data, validation_data, test_data = AisDataLoader._split_dataset(
            self._cfg.train_split, self._cfg.validation_split, self._cfg.test_split, dataset)

        train_loader = DataLoader(train_data, batch_size=self._cfg.batch_size,
                                  shuffle=self._cfg.shuffle,
                                  num_workers=self._cfg.num_workers, collate_fn=AISDatasetMasked.collate_ais_batch)
        validation_loader = DataLoader(validation_data, batch_size=self._cfg.batch_size,
                                       shuffle=self._cfg.shuffle,
                                       num_workers=self._cfg.num_workers, collate_fn=AISDatasetMasked.collate_ais_batch)

        test_loader = DataLoader(test_data, batch_size=self._cfg.batch_size,
                                 shuffle=False,
                                 num_workers=self._cfg.num_workers, collate_fn=AISDatasetMasked.collate_ais_batch)

        return train_loader, validation_loader, test_loader, dataset.stats

    @staticmethod
    def _split_dataset(train_split: float, validation_split: float, test_split: float, dataset: AISDatasetMasked) -> tuple[AISDatasetMasked, AISDatasetMasked, AISDatasetMasked]:
        assert abs(train_split + validation_split + test_split - 1.0) < 1e-6, "Splits must sum to 1"

        indices = np.arange(len(dataset))
        np.random.default_rng(seed=42).shuffle(indices)

        train_size = int(len(indices) * train_split)
        validation_size = int(len(indices) * validation_split)

        train_indices = indices[:train_size]
        validation_indices = indices[train_size:train_size + validation_size]
        test_indices = indices[train_size + validation_size:]

        def extract_subset(idxs):
            return AISDatasetMasked(
                dataset.timestamps[idxs],
                dataset.lats[idxs],
                dataset.lons[idxs],
                dataset.data[idxs],
                dataset.masks[idxs],
                dataset.num_masked_values,
                dataset.num_values_in_sequence,
                dataset.stats,
            )

        return extract_subset(train_indices), extract_subset(validation_indices), extract_subset(test_indices)
