import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import recall_at_fpr
from src.models.baseline_cnn import BaselineCNN


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embeddings, labels = [], []

    for x, y in tqdm(loader, desc="Extracting val embeddings"):
        x = x.to(device)
        z = model(x)
        embeddings.append(z.cpu())
        labels.append(y)

    return torch.cat(embeddings), torch.cat(labels)


def mahalanobis_scores(embeddings, mu, cov_inv):
    diff = embeddings - mu
    return torch.einsum("bi,ij,bj->b", diff, cov_inv, diff)


def select_threshold(val_dataset, normal_model, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    model = BaselineCNN(
        pretrained=cfg.model.pretrained,
        freeze_backbone=True,
    ).to(device)

    model.load_state_dict(normal_model["model_state"])

    embeddings, labels = extract_embeddings(model, loader, device)

    scores = mahalanobis_scores(
        embeddings,
        normal_model["mu"],
        normal_model["cov_inv"],
    )

    result = recall_at_fpr(
        y_true=labels.numpy(),
        scores=scores.numpy(),
        target_fpr=cfg.evaluation.target_fpr,
    )

    return result
