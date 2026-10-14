"""End-to-end smoke test on a tiny synthetic MVTec-style folder (no download needed)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from index_dataset import index_mvtec_dataset  # noqa: E402

from src.training.run import main  # noqa: E402


def _write_images(folder: Path, n: int, rng, defective: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = rng.integers(100, 140, size=(32, 32, 3), dtype=np.uint8)
        if defective:
            img[8:20, 8:20] = 255  # bright square "defect"
        Image.fromarray(img).save(folder / f"{i:03d}.png")


def test_pipeline_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "mvtec"
    _write_images(root / "toy" / "train" / "good", 20, rng, defective=False)
    _write_images(root / "toy" / "test" / "good", 8, rng, defective=False)
    _write_images(root / "toy" / "test" / "scratch", 8, rng, defective=True)

    data_cfg = {
        "dataset": {"root": str(root), "categories": ["toy"]},
        "labels": {"normal_defect_name": "good", "normal_label": 0, "defective_label": 1},
        "splits": {"val_ratio": 0.25, "seed": 0},
    }
    index = index_mvtec_dataset(data_cfg)
    assert set(index.loc[index.split.isin(["train", "val"]), "label"]) == {0}
    index_csv = tmp_path / "index.csv"
    index.to_csv(index_csv, index=False)

    out_dir = tmp_path / "out"
    cfg = {
        "seed": 0,
        "paths": {"dataset_index": str(index_csv), "output_dir": str(out_dir)},
        "data": {"image_size": 32, "categories": ["toy"]},
        "model": {"backbone": "resnet18", "pretrained": False},
        "scoring": {"methods": ["l2_prototype", "mahalanobis", "knn"], "cov_eps": 0.01, "knn_k": 3},
        "evaluation": {"target_fpr": 0.2},
        "runtime": {"batch_size": 8, "num_workers": 0, "device": "cpu"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    assert main(["--config", str(cfg_path)]) == 0

    results = pd.read_csv(out_dir / "results.csv")
    assert list(results["method"]) == ["l2_prototype", "mahalanobis", "knn"]
    assert results[["auroc", "recall", "fpr", "oracle_recall"]].notna().all().all()
    assert (results["val_fpr"] <= 0.2).all()
    for m in results["method"]:
        assert (out_dir / "toy" / f"{m}.pt").exists()
        assert (out_dir / "toy" / f"{m}_test_scores.csv").exists()
