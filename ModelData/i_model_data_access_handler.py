from collections import namedtuple
from abc import ABC, abstractmethod
import datetime

from Types.area import Area

AisMessageTuple = namedtuple('AisMessageTuple', ['timestamp', 'lon', 'lat', 'sog', 'cog', 'vessel_type'])


class IModelDataAccessHandler(ABC):

    @abstractmethod
    def get_ais_messages(self, dates: list[datetime.date], area: Area) -> list[AisMessageTuple]:
        pass
