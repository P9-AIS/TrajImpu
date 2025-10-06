from params import Params
from vec2 import Vec2
from abc import ABC, abstractmethod


class IForceProvider(ABC):

    @abstractmethod
    def get_force(p: Params) -> Vec2:
        pass
