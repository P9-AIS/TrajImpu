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
    northerns: torch.Tensor
    easterns: torch.Tensor


class AISDatasetMasked(Dataset[AISBatch]):
    def __init__(self, timestamps: np.ndarray, northerns: np.ndarray, easterns: np.ndarray, data: np.ndarray, masks: np.ndarray, num_masked_values: int, stats: AISStats):
        self.timestamps = timestamps
        self.northerns = northerns
        self.easterns = easterns
        self.data = data
        self.masks = masks
        self.num_masked_values = num_masked_values
        self.stats = stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> AISBatch:
        return AISBatch(
            observed_data=torch.tensor(self.data[idx], dtype=torch.float32),  # [maxlen, n]
            observed_timestamps=torch.tensor(self.timestamps[idx], dtype=torch.int32, requires_grad=False),  # [maxlen]
            masks=torch.tensor(self.masks[idx], dtype=torch.int8, requires_grad=False),  # [maxlen, n]
            num_missing_values=self.num_masked_values,  # scalar
            northerns=torch.tensor(self.northerns[idx], dtype=torch.float32),
            easterns=torch.tensor(self.easterns[idx], dtype=torch.float32),
        )

    @staticmethod
    def collate_ais_batch(batch):
        return AISBatch(
            observed_data=torch.stack([b.observed_data for b in batch]),
            observed_timestamps=torch.stack([b.observed_timestamps for b in batch]),
            masks=torch.stack([b.masks for b in batch]),
            num_missing_values=max(b.num_missing_values for b in batch),  # or keep as list
            northerns=torch.stack([b.northerns for b in batch]),
            easterns=torch.stack([b.easterns for b in batch]),
        )

    @staticmethod
    def from_ais_dataset_processed(processed_dataset: AISDatasetProcessed, masks: np.ndarray, stats: AISStats) -> "AISDatasetMasked":
        instance = AISDatasetMasked(
            timestamps=processed_dataset.timestamps,
            data=processed_dataset.data,
            masks=masks,
            num_masked_values=stats.num_masked_values,
            stats=stats,
            northerns=processed_dataset.northerns,
            easterns=processed_dataset.easterns,
        )
        return instance

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving masked ais dataset to '{path}'")
        np.savez_compressed(
            path,
            processed_ais_dataset_object=pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL),
            data=self.data,
            timestamps=self.timestamps,
            masks=self.masks,
            northerns=self.northerns,
            easterns=self.easterns,
        )
        print(f"Saved masked ais dataset of {self.data.shape[0]:,} trajectories\n")

    @staticmethod
    def load(path: str) -> "AISDatasetMasked":
        print(f"Loading masked ais dataset from '{path}'")
        with np.load(path, allow_pickle=True) as data:
            dataset: AISDatasetMasked = pickle.loads(data['processed_ais_dataset_object'].item())
            dataset.data = data['data']
            dataset.timestamps = data['timestamps']
            dataset.masks = data['masks']
            dataset.northerns = data['northerns']
            dataset.easterns = data['easterns']
        print(f"Loaded masked ais dataset of {dataset.data.shape[0]:,} trajectories\n")
        return dataset
