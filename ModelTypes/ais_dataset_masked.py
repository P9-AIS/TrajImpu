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
    observed_labels: torch.Tensor
    masks: torch.Tensor
    num_missing_values: int
    num_values_in_sequence: int


class AISDatasetMasked(Dataset[AISBatch]):
    def __init__(self, data: np.ndarray, labels: np.ndarray, masks: np.ndarray, num_masked_values: int, num_values_in_sequence: int, stats: AISStats):
        self.data = data
        self.labels = labels
        self.masks = masks
        self.num_masked_values = num_masked_values
        self.num_values_in_sequence = num_values_in_sequence
        self.stats = stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> AISBatch:
        return AISBatch(
            observed_data=torch.tensor(self.data[idx], dtype=torch.float32),  # [maxlen, n]
            observed_labels=torch.tensor(self.labels[idx], dtype=torch.float32),  # [maxlen, n]
            masks=torch.tensor(self.masks[idx], dtype=torch.int8),  # [maxlen, n]
            num_missing_values=self.num_masked_values,  # scalar
            num_values_in_sequence=self.num_values_in_sequence,  # scalar
        )

    @staticmethod
    def collate_ais_batch(batch):
        return AISBatch(
            observed_data=torch.stack([b.observed_data for b in batch]),
            observed_labels=torch.stack([b.observed_labels for b in batch]),
            masks=torch.stack([b.masks for b in batch]),
            num_missing_values=max(b.num_missing_values for b in batch),  # or keep as list
            num_values_in_sequence=max(b.num_values_in_sequence for b in batch),  # or list
        )

    @staticmethod
    def from_ais_dataset_processed(processed_dataset: AISDatasetProcessed, masks: np.ndarray, num_missing_values: int, num_values_in_sequence: int) -> "AISDatasetMasked":
        instance = AISDatasetMasked(
            data=processed_dataset.data,
            labels=processed_dataset.labels,
            masks=masks,
            num_masked_values=num_missing_values,
            num_values_in_sequence=num_values_in_sequence,
            stats=processed_dataset.stats,
        )
        return instance

    def combine(self, other: "AISDatasetMasked"):
        self.data = np.vstack((self.data, other.data))
        self.labels = np.vstack((self.labels, other.labels))
        self.masks = np.vstack((self.masks, other.masks))
        self.stats = self.stats.combine(other.stats)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving masked ais dataset to '{path}'")
        np.savez_compressed(
            path,
            processed_ais_dataset_object=pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL),
            data=self.data,
            labels=self.labels,
            masks=self.masks,
        )
        print(f"Saved masked ais dataset\n")

    @staticmethod
    def load(path: str) -> "AISDatasetMasked":
        print(f"Loading masked ais dataset from '{path}'")
        with np.load(path, allow_pickle=True) as data:
            dataset: AISDatasetMasked = pickle.loads(data['processed_ais_dataset_object'].item())
            dataset.data = data['data']
            dataset.labels = data['labels']
            dataset.masks = data['masks']
        print(f"Loaded masked ais dataset of size {dataset.data.size}\n")
        return dataset
