from dataclasses import dataclass
import os
import pickle
from torch.utils.data import Dataset
import torch
import numpy as np

from ModelTypes.ais_dataset_processed import AISDatasetProcessed
from ModelTypes.ais_stats import AISStats


@dataclass
class AISBatch:
    observed_data: torch.Tensor
    observed_timestamps: torch.Tensor
    masks: torch.Tensor
    num_missing_values: int
    num_values_in_sequence: int


class AISDatasetMasked(Dataset[AISBatch]):
    def __init__(self, timestamps: np.ndarray, data: np.ndarray, masks: np.ndarray, num_masked_values: int, num_values_in_sequence: int, stats: AISStats):
        self.timestamps = timestamps
        self.data = data
        self.masks = masks
        self.num_masked_values = num_masked_values
        self.num_values_in_sequence = num_values_in_sequence
        self.stats = stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> AISBatch:
        return AISBatch(
            observed_data=torch.tensor(self.data[idx], dtype=torch.float32),  # [maxlen, n]
            observed_timestamps=torch.tensor(self.timestamps[idx], dtype=torch.float32),  # [maxlen]
            masks=torch.tensor(self.masks[idx], dtype=torch.int8),  # [maxlen, n]
            num_missing_values=self.num_masked_values,  # scalar
            num_values_in_sequence=self.num_values_in_sequence,  # scalar
        )

    @staticmethod
    def collate_ais_batch(batch):
        return AISBatch(
            observed_data=torch.stack([b.observed_data for b in batch]),
            observed_timestamps=torch.stack([b.observed_timestamps for b in batch]),
            masks=torch.stack([b.masks for b in batch]),
            num_missing_values=max(b.num_missing_values for b in batch),  # or keep as list
            num_values_in_sequence=max(b.num_values_in_sequence for b in batch),  # or list
        )

    @staticmethod
    def from_ais_dataset_processed(processed_dataset: AISDatasetProcessed, masks: np.ndarray, num_missing_values: int, num_values_in_sequence: int) -> "AISDatasetMasked":
        instance = AISDatasetMasked(
            timestamps=processed_dataset.timestamps,
            data=processed_dataset.data,
            masks=masks,
            num_masked_values=num_missing_values,
            num_values_in_sequence=num_values_in_sequence,
            stats=processed_dataset.stats,
        )
        return instance

    def combine(self, other: "AISDatasetMasked"):
        self.timestamps = np.vstack((self.timestamps, other.timestamps))
        self.data = np.vstack((self.data, other.data))
        self.masks = np.vstack((self.masks, other.masks))
        self.stats = self.stats.combine(other.stats)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving masked ais dataset to '{path}'")
        np.savez_compressed(
            path,
            processed_ais_dataset_object=pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL),
            data=self.data,
            timestamps=self.timestamps,
            masks=self.masks,
        )
        print(f"Saved masked ais dataset\n")

    @staticmethod
    def load(path: str) -> "AISDatasetMasked":
        print(f"Loading masked ais dataset from '{path}'")
        with np.load(path, allow_pickle=True) as data:
            dataset: AISDatasetMasked = pickle.loads(data['processed_ais_dataset_object'].item())
            dataset.data = data['data']
            dataset.timestamps = data['timestamps']
            dataset.masks = data['masks']
        print(f"Loaded masked ais dataset of size {dataset.data.size}\n")
        return dataset
