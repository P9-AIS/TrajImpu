from Config.parser import parse_config
from Connection.postgres_connection import PostgresConnection
from ForceData.force_data_access_handler_db import ForceDataAccessHandlerDb
from ForceProviders.force_provider_traffic import TrafficForceProvider
from ForceUtils.heatmap_generator import generate_heatmap_image


if __name__ == "__main__":
    cfg = parse_config("config.yaml")

    force_data_connection = PostgresConnection(cfg.postgresCfg)
    force_data_handler = ForceDataAccessHandlerDb(force_data_connection)
    force_provider_traffic = TrafficForceProvider(force_data_handler, cfg.trafficForceProviderCfg)

    generate_heatmap_image(force_provider_traffic.get_vectormap(), cfg.heatmapGeneratorCfg)
