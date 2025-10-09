import yaml
from Config.visitor import ConfigVisitorRegistry
from DataAccess.i_data_access_handler import AreaTuple
from DataAccess.postgres_connection import Config as PostgresConfig
from ForceProviders.traffic_force_provider import Config as TrafficForceProviderConfig
from Types.latlon import LatLon
from Utils.heatmap_generator import Config as HeatmapGeneratorConfig
from dataclasses import dataclass


@dataclass
class Config:
    postgresCfg: PostgresConfig
    trafficForceProviderCfg: TrafficForceProviderConfig
    heatmapGeneratorCfg: HeatmapGeneratorConfig


def parse_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return Config(
        postgresCfg=ConfigVisitorRegistry.visit(PostgresConfig, cfg_dict["postgresCfg"]),
        trafficForceProviderCfg=ConfigVisitorRegistry.visit(
            TrafficForceProviderConfig, cfg_dict["trafficForceProviderCfg"]),
        heatmapGeneratorCfg=ConfigVisitorRegistry.visit(HeatmapGeneratorConfig, cfg_dict["heatmapGeneratorCfg"]),
    )


ConfigVisitorRegistry.register(
    PostgresConfig,
    lambda data: PostgresConfig(**data)
)

ConfigVisitorRegistry.register(
    TrafficForceProviderConfig,
    lambda data: TrafficForceProviderConfig(
        start_date=data["start_date"],
        end_date=data["end_date"],
        sample_rate=data["sample_rate"],
        area=ConfigVisitorRegistry.visit(AreaTuple, data["area"]),
        vessel_types=data["vessel_types"],
        base_zoom=data["base_zoom"],
        target_zoom=data["target_zoom"],
        output_dir=data.get("output_dir", "Outputs/Tilemaps")
    )
)

ConfigVisitorRegistry.register(
    HeatmapGeneratorConfig,
    lambda data: HeatmapGeneratorConfig(**data)
)

ConfigVisitorRegistry.register(
    AreaTuple,
    lambda data: AreaTuple(ConfigVisitorRegistry.visit(LatLon, data["bottom_left"]),
                           ConfigVisitorRegistry.visit(LatLon, data["top_right"]))
)

ConfigVisitorRegistry.register(
    LatLon,
    lambda data: LatLon(**data)
)
