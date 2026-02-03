import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.baseline_cnn import BaselineCNN


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embeddings = []

    for x, _ in tqdm(loader, desc="Extracting train embeddings"):
        x = x.to(device)
        z = model(x)  # (B, 512)
        embeddings.append(z.cpu())

    return torch.cat(embeddings, dim=0)


def fit_gaussian(train_dataset, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    model = BaselineCNN(
        pretrained=cfg.model.pretrained,
        freeze_backbone=True,
    ).to(device)

    embeddings = extract_embeddings(model, loader, device)  # (N, 512)

    # Mean
    mu = embeddings.mean(dim=0)

    # Covariance
    X = embeddings - mu
    N = X.shape[0]
    cov = (X.T @ X) / (N - 1)

    # Regularization
    eps = cfg.model.cov_eps
    cov_reg = cov + eps * torch.eye(cov.shape[0])

    cov_inv = torch.linalg.inv(cov_reg)

    return {
        "model_state": model.state_dict(),
        "mu": mu,
        "cov_inv": cov_inv,
    }
