from dataclasses import dataclass
from ModelData.i_model_data_access_handler import IModelDataAccessHandler
from torch.utils.data import DataLoader
from ModelData.i_model_data_access_handler import AisMessageTuple
import random
from ModelUtils.ais_dataset import AISDataset
import torch
from ModelData.i_model_data_access_handler import Config as ModelDataConfig
import os
from itertools import chain


@dataclass
class Config:
    batch_size: int
    shuffle: bool
    num_workers: int
    train_split: float


class AisDataLoader:
    @staticmethod
    def get_data_loaders(cfg: Config, data_handler: IModelDataAccessHandler) -> tuple[DataLoader, DataLoader]:
        train_filepath = f"ModelData/pkl_files/train_loader{AisDataLoader.data_loader_file_name_postfix(cfg, data_handler.config)}.pkl"
        test_filepath = f"ModelData/pkl_files/test_loader{AisDataLoader.data_loader_file_name_postfix(cfg, data_handler.config)}.pkl"

        if os.path.exists(train_filepath) and os.path.exists(test_filepath):
            print("Loading data loaders from files...")
            return torch.load(train_filepath), torch.load(test_filepath)
        else:
            print("Data loaders does not exist. Creating new data loaders...")
            coarse_dataset = data_handler.get_ais_messages()
            print("Grouping data by mmsi...")
            grouped = AisDataLoader.group_dataset(coarse_dataset)
            print("Splitting dataset...")
            train_dataset, test_dataset = AisDataLoader.split_dataset(cfg.train_split, grouped)
            train_data = AisDataLoader.process_dataset(
                train_dataset)
            test_data = AisDataLoader.process_dataset(
                test_dataset)

            train_loader = DataLoader(
                train_data,
                batch_size=cfg.batch_size,
                shuffle=cfg.shuffle,
                num_workers=cfg.num_workers
            )
            test_loader = DataLoader(
                test_data,
                batch_size=cfg.batch_size,
                shuffle=cfg.shuffle,
                num_workers=cfg.num_workers
            )
            print("Saving data loaders to files...")
            # save loads to pkl files
            torch.save(
                train_loader, train_filepath)
            torch.save(
                test_loader, test_filepath)
            return train_loader, test_loader

    @staticmethod
    def data_loader_file_name_postfix(cfg: Config, data_handler_cfg: ModelDataConfig) -> str:
        return f"_bs{cfg.batch_size}_shuf{int(cfg.shuffle)}_nw{cfg.num_workers}_ts{int(cfg.train_split*100)}_" \
            f"ds_{data_handler_cfg.date_start}_{data_handler_cfg.date_end}_area{data_handler_cfg.area}"

    @staticmethod
    def split_dataset(train_split: float, dataset: list[list[AisMessageTuple]]
                      ) -> tuple[list[AisMessageTuple], list[AisMessageTuple]]:

        # Shuffle indices instead of giant list
        idx = list(range(len(dataset)))
        random.shuffle(idx)

        train_size = int(len(idx) * train_split)

        # Select subsets by index
        train_lists = (dataset[i] for i in idx[:train_size])
        test_lists = (dataset[i] for i in idx[train_size:])

        # Flatten using itertools.chain (fast + O(n))
        train_data = list(chain.from_iterable(train_lists))
        test_data = list(chain.from_iterable(test_lists))

        return train_data, test_data

    @staticmethod
    def process_dataset(dataset: list[AisMessageTuple]) -> AISDataset:
        data: list[AisMessageTuple] = []
        labels: list[AisMessageTuple] = []
        masks: list[list[int]] = []
        padding_masks: list[int] = []
        for message in dataset:
            data.append(message)
            labels.append(message)
            masks.append([1 for _ in range(len(message))] if random.random()
                         > 0.1 else [0 for _ in range(len(message))])
            padding_masks.append(1)
        d = AISDataset(data, labels, masks, padding_masks)
        return d

    @staticmethod
    def group_dataset(dataset: list[AisMessageTuple]) -> list[list[AisMessageTuple]]:
        grouped = {}
        for message in dataset:
            if message.mmsi not in grouped:
                grouped[message.mmsi] = []
            grouped[message.mmsi].append(message)
        return list(grouped.values())
