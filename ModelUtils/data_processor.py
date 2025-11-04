from dataclasses import dataclass

from ForceData.i_force_data_access_handler import AisMessageTuple
from Types.ais_dataset_raw import AISDatasetRaw


@dataclass
class Config:
    pass


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def raw_ais_to_dataset(self, raw_ais_messages: list[AisMessageTuple]) -> AISDatasetRaw:
        pass

    def raw_ais_dataset_to_processed(self, raw_ais_dataset: AISDatasetRaw) -> AISDatasetProcessed:
        pass
