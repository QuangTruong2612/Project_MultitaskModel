from src.Project_MultitaskModel import logger
from src.Project_MultitaskModel.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


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