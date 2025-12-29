import numpy as np
import torch
from ForceTypes.params import Params
from ForceTypes.vec3 import Vec3
from abc import ABC, abstractmethod


class IForceProvider(ABC):

    @abstractmethod
    def get_force(self, p: Params) -> Vec3:
        pass

    @abstractmethod
    def get_forces_tensor(self, northerns: torch.Tensor, easterns: torch.Tensor) -> torch.Tensor:
        pass
