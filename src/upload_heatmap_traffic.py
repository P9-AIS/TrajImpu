from Config.parser import parse_config
from Connection.postgres_connection import PostgresConnection
from ForceData.force_data_access_handler_db import ForceDataAccessHandlerDb
from ForceData.force_data_upload_handler_http import ForceDataUploadHandlerHTTP
from ForceProviders.force_provider_traffic import TrafficForceProvider
from ForceUtils.heatmap_generator import generate_heatmap_image


if __name__ == "__main__":
    cfg = parse_config("config.yaml")

    data_handler = ForceDataAccessHandlerDb(PostgresConnection(cfg.postgresCfg))
    traffic_provider = TrafficForceProvider(data_handler, cfg.trafficForceProviderCfg)
    image = generate_heatmap_image(traffic_provider.get_vectormap(), False, cfg.heatmapGeneratorCfg)

    upload_handler = ForceDataUploadHandlerHTTP(cfg.forceDataUploadHandlerCfg)
    upload_handler.upload_image_3034(image, "traffic_heatmap", cfg.trafficForceProviderCfg.area)
