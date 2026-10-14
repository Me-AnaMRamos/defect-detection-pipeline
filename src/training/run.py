"""Fit, calibrate and evaluate every (category, scoring method) pair.

For each category:
  1. extract frozen-backbone embeddings for train / val / test once,
  2. per scoring method: fit on normal train embeddings,
     calibrate the threshold on normal-only val scores at the target FPR,
     evaluate on the test split at that threshold,
  3. save the fitted model, the per-image test scores and a results table.

Usage:
    python -m src.training.run --config configs/train.yaml
    python -m src.training.run --config configs/train.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import AppConfig, load_config
from src.data.dataset import build_datasets
from src.evaluation.metrics import calibrate_threshold, evaluate, metrics_at_threshold
from src.models.backbone import FrozenBackbone, extract_embeddings
from src.scoring import build_scorer
from src.utils import resolve_device, set_seed

REPORT_COLS = ["category", "method", "auroc", "recall", "fpr", "oracle_recall", "val_fpr"]


def embed_category(cfg: AppConfig, category: str, model, device) -> dict:
    datasets = build_datasets(cfg, category)
    out = {}
    for split, ds in datasets.items():
        loader = DataLoader(
            ds,
            batch_size=cfg.runtime.batch_size,
            shuffle=False,
            num_workers=cfg.runtime.num_workers,
        )
        emb, y = extract_embeddings(model, loader, device, desc=f"{category}/{split}")
        out[split] = (emb, y.numpy(), ds.df["image_path"].tolist())
    return out


def run_category(cfg: AppConfig, category: str, model, device) -> list[dict]:
    emb = embed_category(cfg, category, model, device)
    train_x, _, _ = emb["train"]
    val_x, _, _ = emb["val"]
    test_x, test_y, test_paths = emb["test"]

    cat_dir = cfg.paths.output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    target_fpr = cfg.evaluation.target_fpr

    rows = []
    for method in cfg.scoring.methods:
        scorer = build_scorer(method, cfg.scoring.cov_eps, cfg.scoring.knn_k).fit(train_x)

        val_scores = scorer.score(val_x).numpy()
        threshold = calibrate_threshold(val_scores, target_fpr)
        val_fpr = metrics_at_threshold(np.zeros(len(val_scores)), val_scores, threshold)["fpr"]

        test_scores = scorer.score(test_x).numpy()
        result = evaluate(test_y, test_scores, threshold, target_fpr)

        rows.append(
            {
                "category": category,
                "method": method,
                **result,
                "val_fpr": val_fpr,
                "n_train": len(train_x),
                "n_val": len(val_x),
            }
        )

        torch.save(
            {
                "method": method,
                "scorer_state": scorer.state_dict(),
                "threshold": threshold,
                "target_fpr": target_fpr,
                "backbone": cfg.model.backbone,
                "pretrained": cfg.model.pretrained,
                "image_size": cfg.data.image_size,
            },
            cat_dir / f"{method}.pt",
        )
        pd.DataFrame({"image_path": test_paths, "label": test_y, "score": test_scores}).to_csv(
            cat_dir / f"{method}_test_scores.csv", index=False
        )

    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method mean over categories, appended as category='mean'."""
    if df["category"].nunique() < 2:
        return df
    numeric = [
        "auroc",
        "aupr",
        "recall",
        "fpr",
        "precision",
        "oracle_recall",
        "oracle_fpr",
        "val_fpr",
    ]
    mean = df.groupby("method", sort=False)[numeric].mean().reset_index()
    mean.insert(0, "category", "mean")
    return pd.concat([df, mean], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and inputs only")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    if not cfg.paths.dataset_index.exists():
        raise FileNotFoundError(
            f"Dataset index not found at {cfg.paths.dataset_index}. "
            "Run `python data/index_dataset.py` first."
        )
    for category in cfg.data.categories:
        build_datasets(cfg, category)  # fails fast on bad splits

    if args.dry_run:
        print("Config and dataset index OK.")
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    device = resolve_device(cfg.runtime.device)
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = FrozenBackbone(cfg.model.backbone, cfg.model.pretrained).to(device)

    start = time.time()
    rows = []
    for category in cfg.data.categories:
        rows.extend(run_category(cfg, category, model, device))

    df = summarize(pd.DataFrame(rows))
    df.to_csv(out_dir / "results.csv", index=False)
    with (out_dir / "run.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg.to_dict(),
                "device": str(device),
                "torch": torch.__version__,
                "runtime_seconds": round(time.time() - start, 1),
            },
            f,
            indent=2,
        )

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(df[REPORT_COLS].to_string(index=False))
    print(
        f"\nTarget FPR: {cfg.evaluation.target_fpr:.0%}. Results saved to {out_dir / 'results.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
