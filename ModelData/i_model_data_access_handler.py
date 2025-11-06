from abc import ABC, abstractmethod
import datetime as dt
from ModelTypes.ais_dataset_raw import AISDatasetRaw


class IModelDataAccessHandler(ABC):
    @abstractmethod
    def get_ais_messages(self, date: dt.date) -> AISDatasetRaw:
        pass
