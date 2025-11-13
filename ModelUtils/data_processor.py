
import os
from ModelData.i_model_data_access_handler import IModelDataAccessHandler
from ModelTypes.ais_dataset_masked import AISDatasetMasked
from ModelTypes.ais_dataset_processed import AISDatasetProcessed
from ModelTypes.ais_dataset_raw import AISDatasetRaw
import datetime as dt
import numpy as np
from enum import Enum
from dataclasses import dataclass
from ModelUtils.geo_utils import GeoUtils


class MaskingStrategy(Enum):
    POINT_MISSING = 1
    SUB_SEQUENCE_MISSING = 2
    BLOCK_MISSING = 3


@dataclass
class Config:
    traj_len: int = 100
    lead_len: int = 10
    output_dir: str = "Data"
    masking_strategy: MaskingStrategy = MaskingStrategy.POINT_MISSING
    masking_percentage: float = 0.1
    max_time_gap: float = 600.0
    min_traj_gap_distance_m: float = 0.0
    max_traj_gap_distance_m: float = 50.0


class DataProcessor:
    _data_handler: IModelDataAccessHandler
    _cfg: Config
    _rng: np.random.Generator
    _num_masked_values: int

    def __init__(self, data_handler: IModelDataAccessHandler, cfg: Config):
        self._data_handler = data_handler
        self._cfg = cfg
        self._rng = np.random.default_rng()
        self._num_masked_values = int(self._cfg.masking_percentage * self._cfg.traj_len)

        if (self._cfg.lead_len * 2) / self._cfg.traj_len < self._cfg.masking_percentage:
            raise ValueError("Masking percentage is too high for the given lead length and trajectory length.")

    def get_masked_data(self, dates: list[dt.date]) -> AISDatasetMasked:
        print("Getting processed data...")
        processed_data = self._get_processed_data(dates)

        print("Generating masks for processed data...")
        masks = self._get_masks(processed_data)

        masked_data = AISDatasetMasked.from_ais_dataset_processed(
            processed_data, masks, self._num_masked_values, self._cfg.traj_len)

        return masked_data

    def _get_processed_data(self, dates: list[dt.date]) -> AISDatasetProcessed:
        processed_data_file_paths = self._download_processed_data(dates)

        if not processed_data_file_paths:
            raise ValueError("No processed data files were downloaded.")

        dataset = AISDatasetProcessed.load(processed_data_file_paths[0])

        for i in range(1, len(processed_data_file_paths)):
            processed_data = AISDatasetProcessed.load(processed_data_file_paths[i])
            dataset.combine(processed_data)

        return dataset

    def _download_processed_data(self, dates: list[dt.date]) -> list[str]:
        processed_data_file_paths = []

        for date in dates:
            processed_data_file_path = self._get_dataset_filename(date)
            processed_data_file_paths.append(processed_data_file_path)

            if not os.path.exists(processed_data_file_path):
                print("Processed data does not exist. Creating new processed data...")
                coarse_dataset = self._data_handler.get_ais_messages(date)
                processed_data = self._process_dataset(coarse_dataset)
                processed_data.save(processed_data_file_path)

        return processed_data_file_paths

    def _get_dataset_filename(self, date: dt.date) -> str:
        return f"{self._cfg.output_dir}/AISDatasetProcessed/{self._cfg.traj_len=}-date={date}.npz"

    def _process_dataset(self, dataset: AISDatasetRaw) -> AISDatasetProcessed:
        data = self._get_data(dataset)
        # labels = self._get_labels(dataset)
        return AISDatasetProcessed(data, data)

    def _get_data(self, dataset: AISDatasetRaw) -> np.ndarray:
        groups = DataProcessor._group_dataset(dataset)

        if len(groups) == 0:
            raise ValueError("No groups found in dataset.")

        trajectories = self._get_trajectories_from_group(groups[0], self._cfg.traj_len)

        for i in range(1, len(groups)):
            group_trajectories = self._get_trajectories_from_group(groups[i], self._cfg.traj_len)
            if group_trajectories.size > 0:
                trajectories = np.vstack((trajectories, np.array(group_trajectories, dtype=np.float32)))

        return trajectories

    def _get_labels(self, dataset: AISDatasetRaw) -> np.ndarray:
        return np.array([])

    def _get_masks(self, dataset: AISDatasetProcessed) -> np.ndarray:
        num_trajs = dataset.data.shape[0]
        seq_length = dataset.data.shape[1]
        num_attributes = dataset.data.shape[2] - 1  # exclude timestamp

        masks = np.ones((num_trajs, seq_length, num_attributes), dtype=np.int8)

        match self._cfg.masking_strategy:
            case MaskingStrategy.POINT_MISSING:
                for i in range(num_trajs):
                    missing_indices = self._rng.choice(
                        range(self._cfg.lead_len, seq_length - self._cfg.lead_len),
                        self._num_masked_values,
                        replace=False)
                    masks[i, missing_indices, :] = 0.0
            case MaskingStrategy.BLOCK_MISSING:
                for i in range(num_trajs):
                    missing_indices = np.arange(self._num_masked_values) + self._cfg.lead_len + \
                        self._rng.integers(0, seq_length - self._cfg.lead_len - self._num_masked_values)
                    masks[i, missing_indices, :] = 0.0
        return masks

    @staticmethod
    def _group_dataset(dataset: AISDatasetRaw) -> list[AISDatasetRaw]:
        data = dataset.dataset
        mmsi_index = 1
        unique_mmsis = np.unique(data[:, mmsi_index])

        groups: list[AISDatasetRaw] = []
        for mmsi in unique_mmsis:
            group_data = data[data[:, mmsi_index] == mmsi]
            groups.append(AISDatasetRaw(group_data))

        return groups

    def _get_trajectories_from_group(self, group: AISDatasetRaw, traj_len: int) -> np.ndarray:
        data = group.dataset[:, 1:]  # drop mmsi col
        num_attributes = data.shape[1]
        candidate_trajectories = []

        # data_sorted = data[np.argsort(data[:, 0])]

        cur_traj_len = 1
        for i in range(1, data.shape[0]):
            if self._is_trajectory_cut(data[i - 1], data[i]):
                if cur_traj_len >= traj_len:
                    candidate_trajectories.append(data[i - cur_traj_len:i])
                cur_traj_len = 1
            else:
                cur_traj_len += 1

        same_length_trajectories = []

        # overlapping sub-trajectories
        # for traj in candidate_trajectories:
        #     for start_idx in range(0, traj.shape[0] - traj_len + 1):
        #         same_length_trajectories.append(traj[start_idx:start_idx + traj_len])

        # non-overlapping sub-trajectories
        for traj in candidate_trajectories:
            num_full_trajs = traj.shape[0] // traj_len
            for n in range(num_full_trajs):
                start_idx = n * traj_len
                same_length_trajectories.append(traj[start_idx:start_idx + traj_len])

        valid_trajectories = np.ndarray((0, traj_len, num_attributes), dtype=np.float32)
        for traj in same_length_trajectories:
            if self._is_valid_trajectory(traj):
                valid_trajectories = np.vstack((valid_trajectories, np.array([traj], dtype=np.float32)))

        return valid_trajectories

    def _is_trajectory_cut(self, point_1: np.ndarray, point_2: np.ndarray) -> bool:
        time_1 = point_1[0]
        time_2 = point_2[0]
        time_gap = time_2 - time_1

        if time_gap > self._cfg.max_time_gap:
            return True

        loc_1 = point_1[2:4]
        loc_2 = point_2[2:4]
        distance = GeoUtils.haversine_distance_km(loc_1[0], loc_1[1], loc_2[0], loc_2[1]) * 1000.0

        if not (self._cfg.min_traj_gap_distance_m <= distance <= self._cfg.max_traj_gap_distance_m):
            return True

        return False

    def _is_valid_trajectory(self, trajectory: np.ndarray) -> bool:
        return True
