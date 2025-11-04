from dataclasses import dataclass

from Types.ais_dataset_processed import AISDatasetProcessed


@dataclass
class Config:
    pass


class DataLoader:

    @staticmethod
    def get_processed_dataset(cfg: Config) -> AISDatasetProcessed:

        # implementer flow chart her
