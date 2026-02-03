from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecBinaryDataset(Dataset):
    def __init__(
        self,
        index_csv: str,
        split: str,
        transform=None,
        categories: list[str] | None = None,
    ):
        self.df = pd.read_csv(index_csv)

        # Filter split
        self.df = self.df[self.df["split"] == split]

        # Optional category filtering
        if categories is not None:
            self.df = self.df[self.df["category"].isin(categories)]

        if self.df.empty:
            raise ValueError(f"No samples found for split='{split}'")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = Path(row["image_path"])
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def build_datasets(cfg):
    """
    Build train and validation datasets according to config.

    Assumptions:
    - Train split contains ONLY normal samples (label = 0)
    - Validation split may contain both normal and defects
    """

    transform = transforms.Compose(
        [
            transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    index_csv = cfg.paths.dataset_index

    train_dataset = MVTecBinaryDataset(
        index_csv=index_csv,
        split="train",
        transform=transform,
        categories=cfg.data.categories,
    )

    val_dataset = MVTecBinaryDataset(
        index_csv=index_csv,
        split="val",
        transform=transform,
        categories=cfg.data.categories,
    )

    return train_dataset, val_dataset
