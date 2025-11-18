import os
import pickle
import numpy as np

from ForceTypes.vessel_types import VesselType
from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_stats import AISStats


class AISDatasetProcessed():
    def __init__(self, data: np.ndarray):
        assert data.ndim == 3, "Data must be a 3D numpy array (num_samples, seq_len, num_features)."
        self.data, self.stats, self.timestamps = self._get_data(data)

    def combine(self, other: "AISDatasetProcessed"):
        self.data = np.vstack((self.data, other.data))
        self.stats = self.stats.combine(other.stats)

    def _get_data(self, data: np.ndarray) -> tuple[np.ndarray, AISStats, np.ndarray]:
        timestamps = data[:, :, 0]
        data = data[:, :, 1:]
        stats = self._get_stats(data)

        reverse_vessel_type_dict: dict[VesselType, int] = {v: k for k, v in stats.vessel_type_dict.items()}
        vessel_type_column = data[:, :, AISColDict.VESSEL_TYPE.value].astype(int)
        vec_to_enum = np.vectorize(lambda x: VesselType(x))
        vessel_type_enum = vec_to_enum(vessel_type_column)
        vessel_type_indices = np.vectorize(reverse_vessel_type_dict.get)(vessel_type_enum)
        data[:, :, AISColDict.VESSEL_TYPE.value] = vessel_type_indices
        return data, stats, timestamps

    def _get_stats(self, data: np.ndarray) -> AISStats:
        seq_len = data.shape[1]
        num_trajs = data.shape[0]
        num_records = data.shape[0] * data.shape[1]

        lat_column = data[:, :, AISColDict.LATITUDE.value]
        lon_column = data[:, :, AISColDict.LONGITUDE.value]
        vessel_type_column = data[:, :, AISColDict.VESSEL_TYPE.value].astype(int)
        draught_column = data[:, :, AISColDict.DRAUGHT.value]
        sog_column = data[:, :, AISColDict.SOG.value]
        rot_column = data[:, :, AISColDict.ROT.value]

        vessel_types_set = set(np.unique(vessel_type_column))
        vessel_type_dict = {idx: VesselType(v_type) for idx, v_type in enumerate(vessel_types_set)}
        min_lat = np.min(lat_column)
        max_lat = np.max(lat_column)
        min_lon = np.min(lon_column)
        max_lon = np.max(lon_column)
        min_draught = np.min(draught_column)
        max_draught = np.max(draught_column)
        min_sog = np.min(sog_column)
        max_sog = np.max(sog_column)
        min_rot = np.min(rot_column)
        max_rot = np.max(rot_column)

        return AISStats(
            seq_len=seq_len,
            num_trajs=num_trajs,
            num_records=num_records,
            vessel_types=vessel_types_set,
            vessel_type_dict=vessel_type_dict,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
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
