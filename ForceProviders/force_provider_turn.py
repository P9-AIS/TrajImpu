from ForceProviders.i_force_provider import IForceProvider
from ForceTypes.params import Params
from ForceTypes.vec3 import Vec3


class TurnForceProvider(IForceProvider):
    _prev_rot: float = 0.0  # previous rate of turn (°/min)
    _prev_time: int = 0     # previous timestamp in seconds

    def get_force(self, p: Params) -> Vec3:
        cur_rot = p.rot  # current ROT in °/min

        if self._prev_time == 0:
            # first measurement -> no acceleration yet
            angular_accel = 0.0
        else:
            dt = p.time - self._prev_time  # seconds
            angular_accel = (cur_rot - self._prev_rot) / dt / 60.0  # °/s²

        # update previous values
        self._prev_rot = cur_rot
        self._prev_time = p.time

        # return as a Vec3 (optional: store angular acceleration in z-component)
        return Vec3(0.0, 0.0, angular_accel)
