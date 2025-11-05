from collections import namedtuple
from abc import ABC, abstractmethod
import datetime as dt
from dataclasses import dataclass
from Types.area import Area

AisMessageTuple = namedtuple('AisMessageTuple', ['timestamp', 'mmsi',
                             'lat', 'lon', 'rot', 'sog', 'cog', 'heading', 'vessel_type', 'draught'])


@dataclass
class Config:
    date_start: dt.date
    date_end: dt.date
    area: str  # ????????????????? needs to be polygon


class IModelDataAccessHandler(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def get_ais_messages(self) -> list[AisMessageTuple]:
        pass
