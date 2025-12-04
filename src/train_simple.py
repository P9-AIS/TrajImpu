from Config.parser import parse_config
from Model.simple import Model
from ModelData.model_data_access_handler_csv import ModelDataAccessHandlerCSV
from ModelPipeline.trainer import Trainer
from ModelUtils.data_loader import AisDataLoader
from ModelUtils.data_processor import DataProcessor
from ModelUtils.loss_calculator import LossCalculator
from ModelData.model_data_upload_handler_http import ModelDataUploadHandlerHTTP
from ModelData.model_data_upload_handler_mock import ModelDataUploadHandlerMock


if __name__ == "__main__":
    cfg = parse_config("config.yaml")

    # upload_handler = ModelDataUploadHandlerHTTP(cfg.modelDataUploadHandlerCfg)
    upload_handler = ModelDataUploadHandlerMock()

    data_handler = ModelDataAccessHandlerCSV(cfg.modelDataCfg)
    data_processor = DataProcessor(data_handler, cfg.modelDataProcessorCfg)
    data_loader = AisDataLoader(data_processor, cfg.modelDataLoaderCfg)
    train_data_loader, valtest_data_loader, test_data_loader, stats = data_loader.get_data_loaders()
    loss_calculator = LossCalculator()
    model = Model(stats, loss_calculator, cfg.simpleCfg)

    model = model.to(cfg.modelCfg.device)

    trainer = Trainer(model, train_data_loader, valtest_data_loader,
                      test_data_loader, upload_handler, cfg.modelTrainerCfg)
    trainer.train()
