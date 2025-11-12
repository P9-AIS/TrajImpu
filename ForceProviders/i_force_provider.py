import torch
from ForceTypes.params import Params
from ForceTypes.vec3 import Vec3
from abc import ABC, abstractmethod


class IForceProvider(ABC):

    @abstractmethod
    def get_force(self, p: Params) -> Vec3:
        pass

    @abstractmethod
    def get_forces(self, vals: torch.Tensor) -> torch.Tensor:
        pass
