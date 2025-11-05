from dataclasses import dataclass
from ModelData.i_model_data_access_handler import IModelDataAccessHandler
from torch.utils.data import DataLoader
from ModelData.i_model_data_access_handler import AisMessageTuple
import random
from ModelUtils.ais_dataset import AISDataset


@dataclass
class Config:
    batch_size: int
    shuffle: bool
    num_workers: int
    train_split: float


class AisDataLoader:
    @staticmethod
    def get_data_loaders(cfg: Config, data_handler: IModelDataAccessHandler) -> tuple[DataLoader, DataLoader]:
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
        return train_loader, test_loader

    @staticmethod
    def split_dataset(train_split, dataset: list[list[AisMessageTuple]]) -> tuple[list[AisMessageTuple], list[AisMessageTuple]]:
        random.shuffle(dataset)
        train_size = int(len(dataset) * train_split)
        train_data = dataset[:train_size]
        test_data = dataset[train_size:]
        return sum(train_data, []), sum(test_data, [])  # flatten lists

    @staticmethod
    def process_dataset(dataset: list[AisMessageTuple]) -> AISDataset:
        data = []  # list of tuples
        labels = []  # list of tuples
        masks = []  # list of list of bool (one bool per attribute in each message)
        padding_masks = []  # list of bool (one bool per message)
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
