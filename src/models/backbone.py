from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


class FrozenBackbone(nn.Module):
    """ImageNet ResNet18 with the classifier removed: images -> 512-d embeddings."""

    def __init__(self, name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if name != "resnet18":
            raise ValueError(f"Unsupported backbone '{name}'")

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.feature_dim = net.fc.in_features
        net.fc = nn.Identity()

        for p in net.parameters():
            p.requires_grad = False
        self.net = net.eval()

    def train(self, mode: bool = True) -> FrozenBackbone:
        # Keep BatchNorm in inference mode no matter what the caller does.
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def extract_embeddings(
    model: nn.Module, loader: DataLoader, device: torch.device, desc: str = "embeddings"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (embeddings [N, D], labels [N]) for every sample in the loader."""
    model.eval()
    embs, labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        embs.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(embs), torch.cat(labels)
