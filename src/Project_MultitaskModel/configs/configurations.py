from src.Project_MultitaskModel.constants import *
from src.Project_MultitaskModel.utils.common import read_yaml, create_directories
from src.Project_MultitaskModel.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig
from pathlib import Path

class ConfigureManager:
    def __init__(self,
                 config_filepath: Path = CONFIG_FILE_PATH,
                 params_filepath: Path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        
        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params
        create_directories([config.root_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            n_classes=params.N_CLASSES,
            n_segment=params.N_SEGMENT,
            in_channels=params.IN_CHANNELS
        )
        
        return prepare_base_model_config