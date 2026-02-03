import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class BaselineCNN(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()

        # backbone = models.resnet18(pretrained=pretrained)
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)

        # Remove classifier head → feature extractor
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        self.backbone = backbone

    def forward(self, x):
        """
        Returns feature embeddings of shape (B, D)
        """
        return self.backbone(x)
