import os
import pickle
import numpy as np

from ForceTypes.vessel_types import VesselType
from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_stats import AISStats


class AISDatasetProcessed():
    def __init__(self, data: np.ndarray):
        assert data.ndim == 3, "Data must be a 3D numpy array (num_samples, seq_len, num_features)."
        self.data, self.timestamps, self.lats, self.lons = self._get_data(data)

    def combine(self, other: "AISDatasetProcessed"):
        self.data = np.vstack((self.data, other.data))
        self.timestamps = np.vstack((self.timestamps, other.timestamps))
        self.lats = np.vstack((self.lats, other.lats))
        self.lons = np.vstack((self.lons, other.lons))

    def _get_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        timestamps = data[:, :, 0].copy().astype(np.int32)

        lats = data[:, :, 1].copy()
        lons = data[:, :, 2].copy()

        new_data = data[:, :, -2:].copy()

        return new_data, timestamps, lats, lons

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
