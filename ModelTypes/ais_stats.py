from dataclasses import dataclass
from ForceTypes.vessel_types import VesselType


@dataclass
class AISStats:
    seq_len: int
    num_trajs: int
    num_records: int
    vessel_types: set[int]
    vessel_type_dict: dict[int, VesselType]
    min_draught: float
    max_draught: float
    min_sog: float
    max_sog: float
    min_rot: float
    max_rot: float

    def combine(self, other: "AISStats") -> "AISStats":
        combined_num_trajs = self.num_trajs + other.num_trajs
        combined_num_records = self.num_records + other.num_records
        combined_vessel_types = self.vessel_types.union(other.vessel_types)
        combined_vessel_type_dict = self._combine_vessel_type_dicts(other)
        combined_min_draught = min(self.min_draught, other.min_draught)
        combined_max_draught = max(self.max_draught, other.max_draught)
        combined_min_sog = min(self.min_sog, other.min_sog)
        combined_max_sog = max(self.max_sog, other.max_sog)
        combined_min_rot = min(self.min_rot, other.min_rot)
        combined_max_rot = max(self.max_rot, other.max_rot)

        return AISStats(
            seq_len=self.seq_len,
            num_trajs=combined_num_trajs,
            num_records=combined_num_records,
            vessel_types=combined_vessel_types,
            vessel_type_dict=combined_vessel_type_dict,
            min_draught=combined_min_draught,
            max_draught=combined_max_draught,
            min_sog=combined_min_sog,
            max_sog=combined_max_sog,
            min_rot=combined_min_rot,
            max_rot=combined_max_rot
        )

    def _combine_vessel_type_dicts(self, other: "AISStats") -> dict[int, VesselType]:
        dict_a = {v: k for k, v in self.vessel_type_dict.items()}
        dict_b = {v: k for k, v in other.vessel_type_dict.items()}

        keys = set(dict_a.keys()).union(set(dict_b.keys()))
        combined_dict: dict[int, VesselType] = {}

        for key in keys:
            if key in dict_a:
                combined_dict[dict_a[key]] = key
            else:
                combined_dict[dict_b[key]] = key

        return combined_dict
