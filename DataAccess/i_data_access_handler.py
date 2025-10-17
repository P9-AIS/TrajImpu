from collections import namedtuple
from abc import ABC, abstractmethod
import datetime

from Types.area import Area

AisMessageTuple = namedtuple('AisMessageTuple', ['timestamp', 'lon', 'lat', 'sog', 'cog', 'vessel_type'])
DepthTuple = namedtuple('DepthTuple', ['E', 'N', 'depth'])


class IDataAccessHandler(ABC):

    @abstractmethod
    def get_ais_messages_no_stops(self, dates: list[datetime.date], area: Area) -> list[AisMessageTuple]:
        pass

    @abstractmethod
    def get_depths(self, area: Area) -> tuple[int, list[DepthTuple]]:
        pass
