from src.Project_MultitaskModel.constants import *
from src.Project_MultitaskModel.utils.common import read_yaml, create_directories
from src.Project_MultitaskModel.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingModelConfig
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
        
    def get_training_model_config(self) -> TrainingModelConfig:
        training = self.config.training_model
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        create_directories([training.root_dir])
        
        training_model_config = TrainingModelConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            base_model_path=Path(prepare_base_model.base_model_path),
            data_classification=Path(training.data_classification),
            data_segmentation=Path(training.data_segmentation),
            n_classes=params.N_CLASSES,
            n_segment=params.N_SEGMENT,
            in_channels=params.IN_CHANNELS,
            batch_size=params.BATCH_SIZE,
            epochs=params.EPOCHS,
            learning_rate=params.LEARNING_RATE,
            image_size=params.IMAGE_SIZE,
            augmentation=params.AUGMENTATION,
            seed=params.SEED,
            task_num=params.TASK_NUM,
            num_workers=params.NUM_WORKERS,
            img_size=params.IMAGE_SIZE
        )
        
        return training_model_config