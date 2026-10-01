import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torchvision import models

# HAFTA 2: Baseline CNN Modeli
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

# HAFTA 3: ResNet50 Transfer Learning Modeli
class ResNet50Detector(nn.Module):
    def __init__(self, freeze_features=True):
        super(ResNet50Detector, self).__init__()
        # Pretrained ResNet50 ağırlıklarını yüklüyoruz
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # İlk etapta Feature Extractor katmanlarını donduruyoruz (Feature Freezing)
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # Son katmanı (fc) 2 sınıflı (REAL / AI) hale getiriyoruz
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.resnet(x)