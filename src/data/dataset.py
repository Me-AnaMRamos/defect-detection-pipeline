from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


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
