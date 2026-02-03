# src/train.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.models.baseline_cnn import BaselineCNN


# -----------------------
# Transforms
# -----------------------
def get_transforms(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# -----------------------
# Feature extraction
# -----------------------
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embeddings = []

    for x, _ in tqdm(loader, desc="Extracting embeddings"):
        x = x.to(device)
        feats = model(x)  # (B, D)
        embeddings.append(feats.cpu())

    return torch.cat(embeddings, dim=0)


# -----------------------
# Distance-based scoring
# -----------------------
def compute_scores(embeddings, prototype):
    """
    L2 distance to prototype
    """
    return torch.norm(embeddings - prototype, dim=1)


# -----------------------
# Main
# -----------------------
def main(train_dataset, val_dataset, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # Model
    model = BaselineCNN(
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=True,
    ).to(device)

    # -----------------------
    # Train = compute prototype
    # -----------------------
    train_embeddings = extract_embeddings(model, train_loader, device)
    prototype = train_embeddings.mean(dim=0)  # (D,)

    # -----------------------
    # Validation = threshold
    # -----------------------
    val_embeddings = extract_embeddings(model, val_loader, device)
    val_scores = compute_scores(val_embeddings, prototype)

    # Example threshold: 99.5 percentile of good samples
    threshold = torch.quantile(val_scores, 0.995).item()

    print(f"Chosen threshold: {threshold:.4f}")

    # -----------------------
    # Save
    # -----------------------
    torch.save(
        {
            "model": model.state_dict(),
            "prototype": prototype,
            "threshold": threshold,
        },
        f"{cfg['paths']['output_dir']}/baseline_unsupervised.pt",
    )


if __name__ == "__main__":
    pass
