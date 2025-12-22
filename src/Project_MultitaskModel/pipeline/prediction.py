import base64
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.multi_task_model import MultiTaskModelResNet
from PIL import Image
import os
import cv2
import glob
import re
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
        model_dir = 'model'
        files = glob.glob(os.path.join(model_dir, "model_v*.pt"))
        if files:
            # Sắp xếp file dựa trên số phiên bản (regex tìm số sau chữ 'v')
            # Ví dụ: v2 > v1, v10 > v2
            latest_model = max(files, key=lambda f: int(re.search(r'_v(\d+)\.pt', f).group(1)))
            
            print(f"Đang sử dụng model: {latest_model}")
            file_model = latest_model
        else:
            raise FileNotFoundError("Không tìm thấy model nào trong thư mục!")

        model.load_state_dict(torch.load(file_model, map_location=torch.device('cpu')))
        
        imagename = self.filepath
        image = Image.open(imagename).convert('RGB')
        image = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])(image)['image'].unsqueeze(0)
        model.eval()
        
        model.eval()
        with torch.no_grad():
            class_output, seg_output = model(image)
            # SỬA LỖI 1: Lấy trực tiếp index
            pred_index = torch.argmax(class_output, dim=1).item()
            class_name = class_label[pred_index]
        
        mask = torch.sigmoid(seg_output)
        mask = (mask > 0.5).float()
        
        # SỬA LỖI 2: Transpose ảnh về (H, W, C) cho OpenCV
        img_np = image.squeeze().cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0)) # Chuyển (3, 224, 224) -> (224, 224, 3)
        
        # Denormalize nếu cần (vì lúc đầu bạn đã Normalize theo mean/std của ImageNet)
        # Nếu không denormalize, ảnh hiển thị sẽ bị tối hoặc màu kỳ lạ.
        # Công thức: pixel = pixel * std + mean
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1) # Đảm bảo giá trị trong [0, 1]
        
        img_rgb = (img_np * 255).astype(np.uint8)
            
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        
        # Tạo mask màu đỏ
        colored_mask = np.zeros_like(img_rgb)
        colored_mask[:, :, 0] = 255 
        
        # Áp dụng mask lên vùng segmentation
        # mask_np đang là (224, 224), cần mở rộng hoặc dùng làm index
        colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_np)
    
        alpha = 1.0 
        beta = 0.5 
        overlay = cv2.addWeighted(img_rgb, alpha, colored_mask, beta, 0)
        _, buffer = cv2.imencode('.png', overlay)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return [{
            "image": img_base64,       # Trả về ảnh Segmentation
            "class": class_name,       # Trả về tên bệnh
            "confidence": "N/A"        # Có thể thêm xác suất nếu muốn
        }]