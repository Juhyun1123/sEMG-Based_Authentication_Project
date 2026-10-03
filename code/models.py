"""강의 기반 2D CNN 및 사전학습 없는 ResNet18/DenseNet161."""
import torch.nn as nn
from torchvision.models import densenet161, resnet18
from settings import DROPOUT


def create_model(model_name, num_classes=5, dropout=DROPOUT):
    if model_name == "2D_CNN":
        # 3주차의 간단한 2D CNN에 공통 분류기 Dropout을 적용한다.
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )
    if model_name == "ResNet18":
        # 3주차 p.11의 기본 구조를 쓰고 클래스 수만 5로 맞춘다.
        model = resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
        return model
    if model_name == "DenseNet161":
        # 2주차 p.16: 사전학습 없이 DenseNet161과 2208→5 출력층을 사용한다.
        model = densenet161(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model.classifier.in_features, num_classes)
        )
        return model
    raise ValueError(f"지원하지 않는 모델: {model_name}")
