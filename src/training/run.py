from __future__ import annotations

import argparse
import json
import random

import torch

from src.config import load_yaml, parse_config
from src.data.dataset import build_datasets
from src.training.fit_gaussian import fit_gaussian
from src.training.threshold import select_threshold


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config and paths")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    cfg = parse_config(raw)

    set_seed(cfg.seed)

    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("✅ Config loaded. Dry-run OK.")
        print(json.dumps(raw, indent=2))
        print(f"Output dir: {cfg.paths.output_dir.resolve()}")
        return 0

    train_dataset, val_dataset = build_datasets(cfg)

    # 1. Fit normal model
    normal_model = fit_gaussian(train_dataset, cfg)

    # 2. Threshold selection
    threshold_result = select_threshold(val_dataset, normal_model, cfg)

    # 3. Save everything
    torch.save(
        {
            **normal_model,
            "threshold": threshold_result["threshold"],
            "target_fpr": threshold_result["fpr"],
            "recall": threshold_result["recall"],
        },
        cfg.paths.output_dir / "gaussian_ad.pt",
    )

    print("✅ Training complete")
    print(threshold_result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
