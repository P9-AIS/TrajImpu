import os
import pickle
import numpy as np
from ForceUtils.geo_converter import GeoConverter as gc


class AISDatasetProcessed():
    def __init__(self, data: np.ndarray):
        assert data.ndim == 3, "Data must be a 3D numpy array (num_samples, seq_len, num_features)."
        self.data, self.timestamps, self.easterns, self.northerns = self._get_data(data)

    def combine(self, other: "AISDatasetProcessed"):
        self.data = np.vstack((self.data, other.data))
        self.timestamps = np.vstack((self.timestamps, other.timestamps))
        self.easterns = np.vstack((self.easterns, other.easterns))
        self.northerns = np.vstack((self.northerns, other.northerns))

    def _get_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        timestamps = data[:, :, 0].copy().astype(np.int32)

        northerns = data[:, :, -4].copy()
        easterns = data[:, :, -3].copy()
        deltas = data[:, :, -2:].copy()

        return deltas, timestamps, easterns, northerns

    def limit_size(self, max_samples: int):
        if self.data.shape[0] <= max_samples:
            print(f"Dataset already has {self.data.shape[0]} samples <= {max_samples}, no change.")
            return

        print(f"Limiting dataset from {self.data.shape[0]} to {max_samples} samples.")
        self.data = self.data[:max_samples]
        self.timestamps = self.timestamps[:max_samples]
        self.easterns = self.easterns[:max_samples]
        self.northerns = self.northerns[:max_samples]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving processed ais dataset to '{path}'")
        np.savez_compressed(
            path,
            processed_ais_dataset_object=pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL),
            data=self.data,
            timestamps=self.timestamps,
        )
        print(f"Saved processed ais dataset of {self.data.shape[0]:,} trajectories\n")

    @staticmethod
    def load(path: str) -> "AISDatasetProcessed":
        print(f"Loading processed ais dataset from '{path}'")
        with np.load(path, allow_pickle=True) as data:
            dataset: AISDatasetProcessed = pickle.loads(data['processed_ais_dataset_object'].item())
            dataset.data = data['data']
            dataset.timestamps = data['timestamps']
        print(f"Loaded processed ais dataset of {dataset.data.shape[0]:,} trajectories\n")
        return dataset
