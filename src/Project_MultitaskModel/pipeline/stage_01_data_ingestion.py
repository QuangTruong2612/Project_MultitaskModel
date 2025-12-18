

from src.Project_MultitaskModel.components.data_ingestion import DataIngestion
from src.Project_MultitaskModel.configs.configurations import ConfigureManager
from src.Project_MultitaskModel import logger


STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigureManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        # 1. Tải file zip
        data_ingestion.download_file()
        
        # 2. Giải nén
        data_ingestion.extract_zip_file()
            
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e