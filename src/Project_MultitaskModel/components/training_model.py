import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torchvision.models as models
from src.models.multi_task_model import MultiTaskModelResNet
from src.data.loader_data import Data_Seg_Class
from torch.utils.data import DataLoader
from src.loss_func.combined_loss import UncertainlyLoss
from src.metrics import calculate_dice, calculate_iou, EarlyStopping
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.Project_MultitaskModel.entity.config_entity import TrainingModelConfig
from pathlib import Path

class TrainingModel:
    def __init__(self, config: TrainingModelConfig):
        self.config = config
    
    def load_base_model(self):
        model = MultiTaskModelResNet(
            n_classes=self.config.n_classes,
            n_segment=self.config.n_segment,
            in_channels=self.config.in_channels,
            pretrained=False
        )
        state_dict = torch.load(self.config.base_model_path)
        model.load_state_dict(state_dict)
        return model

    def loader_data(self):
        
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            
        img_size = self.config.image_size
        if self.config.augmentation:
            print("Data Augmentation is Enabled")
            transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.GaussNoise(var_limit=(0.01, 0.05), p=0.3),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], 
                is_check_shapes=False )
        else:
            print("Data Augmentation is Disabled")
            transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], 
                is_check_shapes=False )

        train_class_path = os.path.join(self.config.data_classification, 'train')
        train_seg_path = os.path.join(self.config.data_segmentation, 'train')

        data_train = Data_Seg_Class(train_class_path, train_seg_path,  transform=transform)
        train_loader = DataLoader(data_train, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        
        # test_class_path = os.path.join(self.config.data_classification, 'test')
        # test_seg_path = os.path.join(self.config.data_segmentation, 'test')
        # data_test = Data_Seg_Class(test_class_path, test_seg_path, transform=transform)
        # test_loader = DataLoader(data_test, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        
        return train_loader
    
    def train_model(self):
        model = self.load_base_model()
        train_loader = self.loader_data()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = UncertainlyLoss(task_num=self.config.task_num)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        for epoch in range(self.config.epochs):
            model.train()
            running_loss = 0.0
            running_dice = 0.0
            running_iou = 0.0
            correct = 0
            for images, masks, labels in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                outputs_class, outputs_seg = model(images)
                _, pred = torch.max(outputs_class, 1)
                
                loss = criterion(outputs_seg, masks, outputs_class, labels)
                loss.backward()
                optimizer.step()
                
                correct += (pred == labels).sum().item()
                
                running_loss += loss.item()
                dice = calculate_dice(outputs_seg, masks)
                iou = calculate_iou(outputs_seg, masks)
                running_dice += dice.item()
                running_iou += iou.item()

            epoch_loss = running_loss / len(train_loader)
            epoch_dice = running_dice / len(train_loader)
            epoch_iou = running_iou / len(train_loader)
            accuracy = correct / (len(train_loader.dataset))
            print(f'Epoch [{epoch+1}/{self.config.epochs}], Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, IoU: {epoch_iou:.4f}, Accuracy: {accuracy:.4f}')

        self.save_model(model, self.config.trained_model_path)
        print(f'Model saved to {self.config.trained_model_path}')

    @staticmethod
    def save_model(model, path: Path):
        torch.save(model.state_dict(), path)
            