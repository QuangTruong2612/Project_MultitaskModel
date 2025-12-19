import torch
from Project_MultitaskModel.entity.config_entity import EvaluationModelConfig
from Project_MultitaskModel.utils.common import save_json
from models.multi_task_model import MultiTaskModelResNet
from data.loader_data import data_loader
from metrics import calculate_dice, calculate_iou
import dagshub
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse
from pathlib import Path
import os

class Evaluation:
    def __init__(self, config: EvaluationModelConfig):
        self.config = config
    
    def loader_data(self):

        test_class_path = os.path.join(self.config.data_classification, 'test')
        test_seg_path = os.path.join(self.config.data_segmentation, 'test')

        test_loader = data_loader(
            data_classification_path=test_class_path,
            data_segmentation_path=test_seg_path,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            augmentation=self.config.augmentation,
            seed=self.config.seed,
            img_size=self.config.img_size
        )

        return test_loader

    def load_model(self):
        model = MultiTaskModelResNet(
            n_classes=self.config.n_classes,
            n_segment=self.config.n_segment,
            in_channels=self.config.in_channels,
            pretrained=False
        )
        state_dict = torch.load(self.config.trained_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    def evaluation(self):
        self.model = self.load_model()
        test_loader = self.loader_data()
        
        
        self.model.eval()
        with torch.no_grad():
            total_dice = 0.0
            total_iou = 0.0
            correct = 0

            for i, (images, masks, labels) in enumerate(test_loader):
                outputs_class, outputs_seg = self.model(images)

                _, predicted = torch.max(outputs_class, 1)
                correct += (predicted == labels).sum().item()
                dice = calculate_dice(outputs_seg, masks)
                iou = calculate_iou(outputs_seg, masks)
                total_dice += dice.item()
                total_iou += iou.item()
            accuracy = 100 * correct / len(test_loader.dataset)
            avg_dice = total_dice / len(test_loader)
            avg_iou = total_iou / len(test_loader)
        self.scores = {'accuracy': accuracy, 'dice': avg_dice, 'iou': avg_iou}
        
        self.save_score()

    def save_score(self):
        save_json(path=Path("scores.json"), data=self.scores)
    
    def log_into_mlflow(self):
        dagshub.init(repo_owner=str(self.config.repo_owner), repo_name=str(self.config.repo_name))
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.pytorch.log_model(self.model, name="model")
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(self.scores)
            if tracking_url_type_store != "file":
                # Vừa lưu mô hình, vừa ĐĂNG KÝ tên mô hình lên Registry
                mlflow.pytorch.log_model(
                    self.model, 
                    "model", 
                    registered_model_name="MultiTaskModelResNet"
                )
            else:
                # Nếu là file cục bộ, chỉ lưu file mô hình thôi (không đăng ký version)
                mlflow.pytorch.log_model(self.model, "model")
        