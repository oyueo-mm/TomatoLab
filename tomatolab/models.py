"""Model definitions shared by training, evaluation and inference."""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class BasicCNN(nn.Module):
    """Small baseline trained from scratch, intentionally simpler than ResNet18."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def build_model(name: str, num_classes: int, pretrained: bool = False):
    if name == "cnn":
        return BasicCNN(num_classes)
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model '{name}'. Choose cnn or resnet18.")
