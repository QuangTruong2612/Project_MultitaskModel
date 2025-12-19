from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    n_classes: int
    n_segment: int
    in_channels: int
    
@dataclass(frozen=True)
class TrainingModelConfig:
    root_dir: Path
    trained_model_path: Path
    base_model_path: Path
    data_classification: Path
    data_segmentation: Path
    n_classes: int
    n_segment: int
    in_channels: int
    batch_size: int
    epochs: int
    learning_rate: float
    image_size: list
    augmentation: bool
    seed: int
    task_num: int
    num_workers: int
    img_size: int