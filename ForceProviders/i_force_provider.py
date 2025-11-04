import numpy as np
from Types.params import Params
from Types.vec3 import Vec3
from abc import ABC, abstractmethod


class IForceProvider(ABC):

    @abstractmethod
    def get_force(self, p: Params) -> Vec3:
        pass
