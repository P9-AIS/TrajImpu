from abc import ABC, abstractmethod
import datetime as dt
from ModelTypes.ais_dataset_masked import AISDatasetMasked
from ModelTypes.ais_dataset_raw import AISDatasetRaw


class IModelDataUploadHandler(ABC):
    @abstractmethod
    def upload_trajectories(self, dataset: AISDatasetMasked, start_idx: int, end_idx: int) -> None:
        pass
