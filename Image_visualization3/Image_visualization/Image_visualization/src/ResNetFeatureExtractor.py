# 新建一个文件: ResNetFeatureExtractor.py
from turtle import st

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class ResNetFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-2])  # 取前3层特征
        self.model = self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def extract(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(img).cpu().numpy().flatten()
            return features
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None

    def batch_extract(self, path):
        features = []
        valid_images = []
        images = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # 仅处理图片文件

        for img_name in images:
            img_path = os.path.join(path, img_name)
            try:
                feat = self.extract(img_path)
                if feat is not None:
                    features.append(feat)
                    valid_images.append(img_name)
            except Exception as e:
                st.warning(f"无法处理图像 {img_name}: {str(e)}")

        if len(features) == 0:
            raise ValueError("没有有效图像可供处理")

        return np.array(features), valid_images