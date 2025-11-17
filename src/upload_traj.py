from Config.parser import parse_config
from Connection.postgres_connection import PostgresConnection
from ForceData.force_data_access_handler_db import ForceDataAccessHandlerDb
from ForceProviders.force_provider_depth import DepthForceProvider
from ModelData.model_data_upload_handler_http import ModelDataUploadHandlerHTTP
from ModelData.model_data_access_handler_csv import ModelDataAccessHandlerCSV
from ModelUtils.data_processor import DataProcessor
import datetime as dt


if __name__ == "__main__":
    cfg = parse_config("config.yaml")

    data_handler = ModelDataAccessHandlerCSV(cfg.modelDataCfg)
    data_processor = DataProcessor(data_handler, cfg.modelDataProcessorCfg)
    upload_handler = ModelDataUploadHandlerHTTP(cfg.modelDataUploadHandlerCfg)

    masked_data = data_processor.get_masked_data([dt.date(2024, 3, 1)])
    upload_handler.upload_trajectories(masked_data, 0, -1)
