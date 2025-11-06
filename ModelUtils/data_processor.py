
import os
from ModelData.i_model_data_access_handler import IModelDataAccessHandler
from ModelTypes.ais_dataset import AISDatasetProcessed
from ModelTypes.ais_dataset_raw import AISDatasetRaw
import datetime as dt
import numpy as np
from enum import Enum
from dataclasses import dataclass


class MaskingStrategy(Enum):
    POINT_MISSING = 1
    SUB_SEQUENCE_MISSING = 2
    BLOCK_MISSING = 3


@dataclass
class Config:
    min_len: int
    max_len: int
    output_dir: str = "Data"
    masking_strategy: MaskingStrategy = MaskingStrategy.POINT_MISSING
    masking_percentage: float = 0.1


class DataProcessor:
    _data_handler: IModelDataAccessHandler
    _cfg: Config

    def __init__(self, data_handler: IModelDataAccessHandler, cfg: Config):
        self._data_handler = data_handler
        self._cfg = cfg

    def get_processed_data(self, dates: list[dt.date]) -> AISDatasetProcessed:
        processed_data_file_paths = self._download_processed_data(dates)

        dataset = AISDatasetProcessed(np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,)))

        for path in processed_data_file_paths:
            if os.path.exists(path):
                print(f"Loading processed data from file '{path}'...")
                processed_data = AISDatasetProcessed.load(path)
                dataset.combine(processed_data)

        return dataset

    def _download_processed_data(self, dates: list[dt.date]) -> list[str]:
        processed_data_file_paths = []

        for date in dates:
            processed_data_file_path = self._get_dataset_filename(date)
            processed_data_file_paths.append(processed_data_file_path)

            if os.path.exists(processed_data_file_path):
                print("Loading processed data from file...")
                processed_data = AISDatasetProcessed.load(processed_data_file_path)
            else:
                print("Processed data does not exist. Creating new processed data...")
                coarse_dataset = self._data_handler.get_ais_messages(date)
                processed_data = self._process_dataset(coarse_dataset)
                processed_data.save(processed_data_file_path)

        return processed_data_file_paths

    def _get_dataset_filename(self, date: dt.date) -> str:
        return f"{self._cfg.output_dir}/AISDatasetProcessed/{self._cfg.min_len=}-{self._cfg.max_len=}-date={date}.pkl"

    def _process_dataset(self, dataset: AISDatasetRaw) -> AISDatasetProcessed:
        data = self._get_data(dataset)
        labels = self._get_labels(dataset)
        masks = self._get_masks(dataset)
        padding_masks = self._get_padding_masks(dataset)

        d = AISDatasetProcessed(data, labels, masks, padding_masks)
        return d

    def _get_data(self, dataset: AISDatasetRaw) -> np.ndarray:
        match self._cfg.masking_strategy:
            case MaskingStrategy.POINT_MISSING:
                pass
            case MaskingStrategy.SUB_SEQUENCE_MISSING:
                pass
            case MaskingStrategy.BLOCK_MISSING:
                pass
        return np.array([])

    def _get_labels(self, dataset: AISDatasetRaw) -> np.ndarray:
        match self._cfg.masking_strategy:
            case MaskingStrategy.POINT_MISSING:
                pass
            case MaskingStrategy.SUB_SEQUENCE_MISSING:
                pass
            case MaskingStrategy.BLOCK_MISSING:
                pass
        return np.array([])

    def _get_masks(self, dataset: AISDatasetRaw) -> np.ndarray:
        match self._cfg.masking_strategy:
            case MaskingStrategy.POINT_MISSING:
                pass
            case MaskingStrategy.SUB_SEQUENCE_MISSING:
                pass
            case MaskingStrategy.BLOCK_MISSING:
                pass
        return np.array([])

    def _get_padding_masks(self, dataset: AISDatasetRaw) -> np.ndarray:
        match self._cfg.masking_strategy:
            case MaskingStrategy.POINT_MISSING:
                pass
            case MaskingStrategy.SUB_SEQUENCE_MISSING:
                pass
            case MaskingStrategy.BLOCK_MISSING:
                pass
        return np.array([])

    @staticmethod
    def _group_dataset(dataset: AISDatasetRaw) -> list[AISDatasetRaw]:
        # grouped = {}
        # for message in dataset:
        #     if message.mmsi not in grouped:
        #         grouped[message.mmsi] = []
        #     grouped[message.mmsi].append(message)
        # return list(grouped.values())
        return []
