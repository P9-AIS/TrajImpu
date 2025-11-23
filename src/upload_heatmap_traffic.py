from Config.parser import parse_config
from Connection.postgres_connection import PostgresConnection
from ForceData.force_data_access_handler_db import ForceDataAccessHandlerDb
from ForceData.force_data_upload_handler_http import ForceDataUploadHandlerHTTP
from ForceProviders.force_provider_traffic import TrafficForceProvider
from ForceUtils.heatmap_generator import generate_heatmap_image
from ForceUtils.reprojector import Reprojector

if __name__ == "__main__":
    cfg = parse_config("config.yaml")

    data_handler = ForceDataAccessHandlerDb(PostgresConnection(cfg.postgresCfg))
    traffic_provider = TrafficForceProvider(data_handler, cfg.trafficForceProviderCfg)
    image_path_3034 = generate_heatmap_image(traffic_provider.get_vectormap(), cfg.heatmapGeneratorCfg)
    image_path_3857 = image_path_3034.replace(".png", "_3857.png")

    new_area = Reprojector.reproject_png_3034_to_3857(
        image_path_3034,
        image_path_3857,
        cfg.trafficForceProviderCfg.area
    )

    upload_handler = ForceDataUploadHandlerHTTP(cfg.forceDataUploadHandlerCfg)
    upload_handler.upload_image(image_path_3034, "traffic_heatmap_3034", cfg.trafficForceProviderCfg.area)
    upload_handler.upload_image(image_path_3857, "traffic_heatmap_3857", new_area)
