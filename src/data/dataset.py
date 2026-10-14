from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SPLITS = ("train", "val", "test")


class MVTecBinaryDataset(Dataset):
    """Rows of the dataset index for one split (and optionally some categories)."""

    def __init__(
        self,
        index_csv: str | Path,
        split: str,
        transform=None,
        categories: list[str] | tuple[str, ...] | None = None,
    ):
        df = pd.read_csv(index_csv)
        df = df[df["split"] == split]
        if categories is not None:
            df = df[df["category"].isin(list(categories))]
        if df.empty:
            raise ValueError(f"No samples for split='{split}', categories={categories}")

        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_datasets(cfg, category: str) -> dict[str, MVTecBinaryDataset]:
    """Train / val / test datasets for one category.

    - train: normal only (fit the normality model)
    - val:   normal only (calibrate the threshold)
    - test:  normal + defective (report performance)
    """
    tf = eval_transform(cfg.data.image_size)
    datasets = {
        split: MVTecBinaryDataset(cfg.paths.dataset_index, split, tf, categories=[category])
        for split in SPLITS
    }
    for split in ("train", "val"):
        if (datasets[split].df["label"] != 0).any():
            raise ValueError(f"'{split}' split for '{category}' must contain only normal images")
    if datasets["test"].df["label"].nunique() < 2:
        raise ValueError(f"'test' split for '{category}' needs normal and defective images")
    return datasets
