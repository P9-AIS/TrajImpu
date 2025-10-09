from collections import namedtuple
from abc import ABC, abstractmethod
import datetime

AisMessageTuple = namedtuple('AisMessageTuple', ['timestamp', 'lat', 'lon', 'sog', 'cog', 'vessel_type'])
AreaTuple = namedtuple('AreaTuple', ['bot_left', 'top_right'])


class IDataAccessHandler(ABC):

    @abstractmethod
    def get_ais_messages(self, dates: list[datetime.date], area: AreaTuple) -> list[AisMessageTuple]:
        pass
