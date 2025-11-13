import math

import torch


class GeoUtils:

    @staticmethod
    def haversine_distance_km(lat1, lon1, lat2, lon2):
        # Convert decimal degrees to radians
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371.0  # Radius of Earth in kilometers
        distance = r * c

        return distance

    @staticmethod
    def haversine_distances_m(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        R = 6371000.0  # Radius of Earth in meters

        lat1 = torch.deg2rad(pos1[..., 0])
        lon1 = torch.deg2rad(pos1[..., 1])
        lat2 = torch.deg2rad(pos2[..., 0])
        lon2 = torch.deg2rad(pos2[..., 1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a))

        distance = R * c
        return distance
