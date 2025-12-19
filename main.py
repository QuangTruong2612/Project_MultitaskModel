from src.Project_MultitaskModel import logger
from src.Project_MultitaskModel.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Project_MultitaskModel.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.Project_MultitaskModel.pipeline.stage_03_training_model import TrainingModelPipeline

logger.info("Project_MultitaskModel package initialized.")

STAGE_NAME = "Data Ingestion Stage" 
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training Model Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = TrainingModelPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e