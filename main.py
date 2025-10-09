from Config.parser import parse_config
from DataAccess.data_access_handler import DataAccessHandler
from ForceProviders.traffic_force_provider import TrafficForceProvider
from Utils.heatmap_generator import generate_heatmap_image
from DataAccess.postgres_connection import PostgresConnection


def main():
    cfg = parse_config("config.yaml")

    prov = TrafficForceProvider(DataAccessHandler(PostgresConnection(cfg.postgresCfg)), cfg.trafficForceProviderCfg)

    generate_heatmap_image(prov._vectormap, cfg.heatmapGeneratorCfg)


if __name__ == "__main__":
    main()
