import os
import pickle
import numpy as np

from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_stats import AISStats


class AISDatasetProcessed():
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        assert labels.ndim == 3, "Labels must be a 3D numpy array (num_samples, seq_len, num_features)."
        assert data.ndim == 3, "Data must be a 3D numpy array (num_samples, seq_len, num_features)."
        assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples."
        assert data.shape[1] == labels.shape[1], "Data and labels must have the same sequence length."
        assert data.shape[2] == labels.shape[2], "Data and labels must have the same number of features."

        self.data = data
        self.labels = labels
        self.stats = self._get_stats()

    def combine(self, other: "AISDatasetProcessed"):
        self.data = np.vstack((self.data, other.data))
        self.labels = np.vstack((self.labels, other.labels))
        self.stats = self.stats.combine(other.stats)

    def _get_stats(self) -> AISStats:
        num_trajs = self.data.shape[0]
        num_records = self.data.shape[0] * self.data.shape[1]

        vessel_type_column = self.data[:, :, AISColDict.VESSEL_TYPE.value].astype(int)
        draught_column = self.data[:, :, AISColDict.DRAUGHT.value]
        sog_column = self.data[:, :, AISColDict.SOG.value]
        rot_column = self.data[:, :, AISColDict.ROT.value]

        vessel_types_set = set(np.unique(vessel_type_column))
        min_draught = np.min(draught_column)
        max_draught = np.max(draught_column)
        min_sog = np.min(sog_column)
        max_sog = np.max(sog_column)
        min_rot = np.min(rot_column)
        max_rot = np.max(rot_column)

        return AISStats(
            num_attributes=self.data.shape[2],
            num_trajs=num_trajs,
            num_records=num_records,
            vessel_types=vessel_types_set,
            min_draught=min_draught,
            max_draught=max_draught,
            min_sog=min_sog,
            max_sog=max_sog,
            min_rot=min_rot,
            max_rot=max_rot
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving processed ais dataset to '{path}'")
        np.savez_compressed(
            path,
            processed_ais_dataset_object=pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL),
            data=self.data,
            labels=self.labels,
        )
        print(f"Saved processed ais dataset\n")

    @staticmethod
    def load(path: str) -> "AISDatasetProcessed":
        print(f"Loading processed ais dataset from '{path}'")
        with np.load(path, allow_pickle=True) as data:
            dataset: AISDatasetProcessed = pickle.loads(data['processed_ais_dataset_object'].item())
            dataset.data = data['data']
            dataset.labels = data['labels']
        print(f"Loaded processed ais dataset of size {dataset.data.size}\n")
        return dataset
