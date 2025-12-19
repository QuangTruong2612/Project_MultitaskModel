import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.multi_task_model import MultiTaskModelResNet
from PIL import Image
import os
import cv2

class PredictPipeline:
    def __init__(self, filepath):
        self.filepath = filepath
        
    def predict(self):
        class_label={0: 'Glioma', 1: 'Meningioma', 2: 'No-tumor', 3: 'Pituitary'}
        model = MultiTaskModelResNet(
            n_classes=4,
            n_segment=1,
            in_channels=3,
            pretrained=False
        )
        file_model = os.path.join("model", "model.pt")
        model.load_state_dict(torch.load(file_model,  map_location=torch.device('cpu')))
        
        imagename = self.filepath
        image = Image.open(imagename).convert('RGB')
        image = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])(image)['image'].unsqueeze(0)
        model.eval()
        
        with torch.no_grad():
            class_output, seg_output = model(image)
            class_pred = torch.argmax(class_output, dim=1).item()
            
        for class_pred in class_label:
            class_name = class_label[class_pred]
        
        mask = torch.sigmoid(seg_output)
        mask = (mask > 0.5).float()
        
        img_np = image.squeeze().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        if len(img_np.shape) == 2: # Nếu là ảnh xám
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_np
            
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        colored_mask = np.zeros_like(img_rgb)
        colored_mask[:, :, 0] = 255 # Kênh Red = 255
        colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_np)
    
        # Trộn ảnh gốc và mask màu với độ trong suốt (Alpha blending)
        # công thức: output = image * alpha + mask * beta + gamma
        alpha = 1.0   # Độ đậm ảnh gốc
        beta = 0.5    # Độ đậm mask (0.5 là bán trong suốt)
        overlay = cv2.addWeighted(img_rgb, alpha, colored_mask, beta, 0)
        
        return [{
            "predicted_class": class_name,
            "segmentation_mask": overlay
            }]