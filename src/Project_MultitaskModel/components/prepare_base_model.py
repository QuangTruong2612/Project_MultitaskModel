import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torchvision.models as models
from src.models.multi_task_model import MultiTaskModelResNet
from src.Project_MultitaskModel.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        
    def get_base_model(self):
        self.model = MultiTaskModelResNet(
            n_classes=self.config.n_classes,
            n_segment=self.config.n_segment,
            in_channels=self.config.in_channels
            )

        self.save_model(self.model, self.config.base_model_path)
    
    @staticmethod
    def save_model(model: torch.nn.Module, path: Path):
        torch.save(model.state_dict(), path)
        