from Config.parser import parse_config
from DataAccess.i_data_access_handler import AreaTuple
from ForceProviders.traffic_force_provider import (
    TrafficForceProvider,
    Config as TFPConfig,
)
import datetime as dt
from Types.latlon import LatLon
from Utils.heatmap_generator import generate_heatmap_image
from DataAccess.mock_data_access_handler import MockDataAccessHandler
from DataAccess.postgres_connection import PostgresConnection, Config as PGCfg


def main():
    cfg = parse_config("config.yaml")

    prov = TrafficForceProvider(cfg.trafficForceProviderCfg, data_handler=PostgresConnection(cfg.postgresCfg))

    generate_heatmap_image(prov._vectormap, cfg.heatmapGeneratorCfg)


if __name__ == "__main__":
    main()
