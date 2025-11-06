from ForceProviders.i_force_provider import IForceProvider
from ForceTypes.params import Params
from ForceTypes.vec3 import Vec3
import datetime as dt
from ForceUtils.geo_converter import GeoConverter as gc


class PropulsionForceProvider(IForceProvider):
    _prev_vel: Vec3 = Vec3(0.0, 0.0, 0.0)
    _prev_time: int = 0

    def get_force(self, p: Params) -> Vec3:
        heading_vec = gc.espg4326_heading_to_espg3034_heading(p.cog, p.lon, p.lat, dt.date.fromtimestamp(p.time))
        cur_vel = heading_vec * p.sog

        accel_vec = (cur_vel - self._prev_vel) / (p.time - self._prev_time)
        self._prev_vel = cur_vel
        self._prev_time = p.time
        return accel_vec
