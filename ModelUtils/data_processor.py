
import math
import os

import pyproj
from tqdm import tqdm
from ForceProviders.i_force_provider import IForceProvider
from ModelData.i_model_data_access_handler import IModelDataAccessHandler
from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_dataset_masked import AISDatasetMasked
from ModelTypes.ais_dataset_processed import AISDatasetProcessed
from ModelTypes.ais_dataset_raw import AISDatasetRaw
import datetime as dt
import numpy as np
from enum import Enum
from dataclasses import dataclass
from ModelUtils.geo_utils import GeoUtils
from ForceUtils.geo_converter import GeoConverter as gc


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
    min_sog: float = 1
    max_time_gap: float = 600.0
    max_traj_gap_distance_m: float = 50.0


class DataProcessor:
    _data_handler: IModelDataAccessHandler
    _cfg: Config
    _rng: np.random.Generator
    _num_masked_values: int
    _depth_force_provider: IForceProvider

    def __init__(self, data_handler: IModelDataAccessHandler, depth_force_provider: IForceProvider, cfg: Config):
        self._data_handler = data_handler
        self._depth_force_provider = depth_force_provider
        self._cfg = cfg
        self._rng = np.random.default_rng()
        self._num_masked_values = int(self._cfg.masking_percentage * self._cfg.traj_len)

        if self._num_masked_values % 2 != 0:
            self._num_masked_values += 1

        if self._cfg.masking_percentage * self._cfg.traj_len >= (self._cfg.traj_len - 2 * self._cfg.lead_len):
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
        return f"{self._cfg.output_dir}/AISDatasetProcessed/{self._cfg.traj_len=}-{self._cfg.lead_len=}-{self._cfg.min_sog=}-{self._cfg.max_time_gap=}-{self._cfg.max_traj_gap_distance_m=}-date={date}.npz"

    def _process_dataset(self, dataset: AISDatasetRaw) -> AISDatasetProcessed:
        data = self._get_data(dataset)
        return AISDatasetProcessed(data)

    def _get_data(self, dataset: AISDatasetRaw) -> np.ndarray:
        groups = DataProcessor._group_dataset(dataset)

        if len(groups) == 0:
            raise ValueError("No groups found in dataset.")

        initial_trajectories = []

        for i in tqdm(range(len(groups)), desc="Processing trajectory groups"):
            group_trajectories = self._get_trajectories_from_group(groups[i], self._cfg.traj_len)
            if group_trajectories:
                initial_trajectories.extend(group_trajectories)

        filtered_trajectories = self._filter_trajectories_forces(initial_trajectories, self._depth_force_provider)

        trajectories = np.array(filtered_trajectories, dtype=np.float64)

        delta_E, delta_N = DataProcessor.get_deltas(trajectories)

        trajectories = np.concatenate(
            (trajectories, delta_N[:, :, np.newaxis], delta_E[:, :, np.newaxis]), axis=2, dtype=np.float64)

        return trajectories

    def _get_masks(self, dataset: AISDatasetProcessed) -> np.ndarray:
        num_trajs = dataset.data.shape[0]
        seq_length = dataset.data.shape[1]
        num_attributes = dataset.data.shape[2]

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
    def get_deltas(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch, seq, _ = trajectories.shape

        # Initialize pyproj transformer: WGS84 lat/lon -> EPSG:3034 (meters)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)

        # Convert all lat/lon to E/N in meters
        lons = trajectories[:, :, 2]
        lats = trajectories[:, :, 1]

        E, N = transformer.transform(lons.reshape(-1), lats.reshape(-1))
        E = E.reshape(batch, seq)
        N = N.reshape(batch, seq)

        # Compute step-wise deltas in meters
        delta_E = np.zeros_like(E)
        delta_N = np.zeros_like(N)
        delta_E[:, 1:] = E[:, 1:] - E[:, :-1]
        delta_N[:, 1:] = N[:, 1:] - N[:, :-1]

        return delta_E, delta_N

    @staticmethod
    def _spatially_convert_dataset(trajectories: np.ndarray) -> np.ndarray:
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)
        batch, seq, _ = trajectories.shape

        lons = trajectories[:, :, 2].reshape(-1)
        lats = trajectories[:, :, 1].reshape(-1)
        E, N = transformer.transform(lons, lats)
        E = E.reshape(batch, seq)
        N = N.reshape(batch, seq)

        trajectories[:, :, 1] = N  # latitude → north
        trajectories[:, :, 2] = E  # longitude → east

        return trajectories

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

    def _get_trajectories_from_group(self, group: AISDatasetRaw, traj_len: int) -> list[np.ndarray]:
        data = np.delete(group.dataset, 1, axis=1)  # drop mmsi col
        num_attributes = data.shape[1]
        candidate_trajectories = []

        cur_traj_len = 1
        for i in range(1, data.shape[0]):
            if self._is_trajectory_cut(data[i - 1], data[i]):
                if cur_traj_len >= traj_len:
                    candidate_trajectories.append(data[i - cur_traj_len:i])
                cur_traj_len = 1
            else:
                cur_traj_len += 1

        same_length_trajectories = []

        for traj in candidate_trajectories:
            num_full_trajs = traj.shape[0] // traj_len
            for n in range(num_full_trajs):
                start_idx = n * traj_len
                same_length_trajectories.append(traj[start_idx:start_idx + traj_len])

        valid_trajectories = [traj for traj in same_length_trajectories if self._is_valid_trajectory(traj)]

        return valid_trajectories

    def _is_trajectory_cut(self, point_1: np.ndarray, point_2: np.ndarray) -> bool:
        time_1 = point_1[0]
        time_2 = point_2[0]
        time_gap = time_2 - time_1

        if time_gap > self._cfg.max_time_gap:
            return True

        loc_1 = point_1[1:3]
        loc_2 = point_2[1:3]
        distance = GeoUtils.haversine_distance_km(loc_1[0], loc_1[1], loc_2[0], loc_2[1]) * 1000.0

        sog_1 = point_1[4]
        sog_2 = point_2[4]

        if distance > self._cfg.max_traj_gap_distance_m or sog_1 < 1 or sog_2 < 1:
            return True

        return False

    def _is_valid_trajectory(self, trajectory: np.ndarray) -> bool:
        return True

    def _filter_trajectories_forces(self, trajectories: list[np.ndarray], force_provider: IForceProvider) -> list[np.ndarray]:
        traj_and_impacts = [(traj, DataProcessor._get_trajectory_perpendicular_impact(traj, force_provider))
                            for traj in trajectories]

        # Here we sort descending (highest impact first)
        traj_and_impacts.sort(key=lambda x: x[1], reverse=True)

        n = len(traj_and_impacts)
        if n == 0:
            return []

        cutoff_start = int(n * 0.01)
        cutoff_end = int(n * 0.05)

        start_index = cutoff_start
        end_index = cutoff_end

        if start_index >= end_index:
            return trajectories

        filtered_pairs = traj_and_impacts[start_index:end_index]

        return [pair[0] for pair in filtered_pairs]

    def _filter_trajectories_curvature(self, trajectories: list[np.ndarray]) -> list[np.ndarray]:
        filtered = []

        # --- CONFIGURATION ---
        # 1. Smoothness Limit: No single turn can be sharper than this (e.g., 45 degrees)
        MAX_ALLOWED_SHARPNESS = 75.0

        # 2. Minimum Curviness: The path must turn at least this much in total (e.g., 20 degrees)
        MIN_REQUIRED_CURVATURE = 90
        # ---------------------

        for traj in trajectories:
            max_turn, total_curve = DataProcessor.get_trajectory_metrics(traj)

            # LOGIC:
            # It must be smooth (max_turn low)
            # AND
            # It must actually curve (total_curve high)
            if max_turn < MAX_ALLOWED_SHARPNESS and total_curve > MIN_REQUIRED_CURVATURE:
                filtered.append(traj)

        return filtered

    @staticmethod
    def _get_trajectory_force_impact(trajectory: np.ndarray, force_provider: IForceProvider) -> np.ndarray:
        lats = trajectory[:, 1]
        lons = trajectory[:, 2]

        forces = force_provider.get_forces_np(np.stack((lats, lons), axis=-1)[np.newaxis, :, :])

        force_magnitudes = np.hypot(forces[0, :, 0], forces[0, :, 1])
        force_impact = np.sum(force_magnitudes)

        return force_impact

    @staticmethod
    def _get_trajectory_perpendicular_impact(trajectory: np.ndarray, force_provider: IForceProvider) -> float:
        # 1. Get Trajectory Direction Vectors (The ship's movement)
        # Note: We need vectors for each point. Since we have N points, we get N-1 segments.
        # We can repeat the last vector to match the shape or discard the last force point.
        # Let's discard the last force point to match the N-1 movement segments.

        # Extract Lat/Lon (assuming cols 1 and 2)
        coords = trajectory[:, 1:3]

        # Calculate displacement vectors (d_lat, d_lon)
        # shape: (N-1, 2)
        move_vecs = np.diff(coords, axis=0)

        # Normalize movement vectors to get pure Direction (Unit Vectors)
        move_norms = np.linalg.norm(move_vecs, axis=1, keepdims=True)
        # Avoid division by zero for stationary points
        move_norms[move_norms < 1e-6] = 1.0
        move_dir = move_vecs / move_norms

        # 2. Get Forces at these points
        # We only care about forces at the start of each segment (points 0 to N-1)
        lats = coords[:-1, 0]
        lons = coords[:-1, 1]

        # force_provider expects (Batch, Steps, 2). We give it 1 batch.
        # forces shape: (1, N-1, 2) -> we take [0] to get (N-1, 2)
        forces = force_provider.get_forces_np(np.stack((lats, lons), axis=-1)[np.newaxis, :, :])[0]

        # 3. Calculate Perpendicular Component (The "Sideways" Force)
        # 2D Cross Product (determinant):  A_x * B_y - A_y * B_x
        # move_dir is A, forces is B
        # Result is the scalar magnitude of the force acting perpendicular to movement
        perp_forces = move_dir[:, 0] * forces[:, 1] - move_dir[:, 1] * forces[:, 0]

        # 4. Summarize
        # We take absolute value because we care if it pushes Left OR Right
        total_perp_impact = np.sum(np.abs(perp_forces))

        return float(total_perp_impact)

    @staticmethod
    def get_trajectory_metrics(trajectory: np.ndarray):
        # 1. Get the displacement vectors in meters (The "Deltas")
        # delta_E/N shape is expected to be (Batch, Steps). We take index [0].
        # These represent the vector of movement for each time step.
        delta_E, delta_N = DataProcessor.get_deltas(trajectory.reshape(1, trajectory.shape[0], trajectory.shape[1]))

        # Stack them to get shape (Num_Segments, 2)
        # vectors[i] = [meters_east, meters_north] for step i
        vectors = np.stack((delta_E[0], delta_N[0]), axis=-1)

        # 2. Normalize to get Unit Vectors (Direction only)
        # Calculate speed/distance of each segment
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        # --- CRITICAL FILTERING ---
        # If the ship is standing still (norm ~ 0), it has no "direction".
        # Comparing a "stopped" vector to a "moving" vector results in garbage angles.
        # We filter out segments where movement is less than 1mm (noise/stationary).
        moving_mask = norms.flatten() > 1e-3

        if np.sum(moving_mask) < 2:
            # Not enough moving segments to calculate a turn
            return 0.0, 0.0

        valid_vectors = vectors[moving_mask]
        valid_norms = norms[moving_mask].reshape(-1, 1)

        # Divide by norm to get unit vectors (length 1.0)
        unit_vectors = valid_vectors / valid_norms

        # 3. Calculate angles between CONSECUTIVE segments
        # We dot product segment[i] with segment[i+1]
        # dot(a, b) = |a||b|cos(theta) -> since length is 1, dot = cos(theta)
        dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)

        # Clip to [-1.0, 1.0] to prevent NaN errors from floating point precision
        dot_products = np.clip(dot_products, -1.0, 1.0)

        # 4. Convert to Degrees
        angles_deg = np.degrees(np.arccos(dot_products))

        # 5. Compute Metrics
        if len(angles_deg) == 0:
            return 0.0, 0.0

        max_single_turn = np.max(angles_deg)      # "Sharpness" (e.g., did it snap 90 degrees?)
        total_curvature = np.sum(angles_deg)      # "Curviness" (how much did it wind overall?)

        return max_single_turn, total_curvature
