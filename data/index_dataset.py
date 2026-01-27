import os
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))


def index_mvtec_dataset(cfg: dict) -> pd.DataFrame:
    root = Path(cfg["dataset"]["root"])
    categories = cfg["dataset"]["categories"]

    normal_defect = cfg["labels"]["normal_defect_name"]
    normal_label = cfg["labels"]["normal_label"]
    defective_label = cfg["labels"]["defective_label"]

    records = []

    for category in categories:
        category_path = root / category
        if not category_path.exists():
            raise FileNotFoundError(f"Category not found: {category_path}")

        for split in ["train", "test"]:
            split_path = category_path / split
            if not split_path.exists():
                continue

            for defect_type in os.listdir(split_path):
                defect_path = split_path / defect_type
                if not defect_path.is_dir():
                    continue

                label = normal_label if defect_type == normal_defect else defective_label

                for img_name in os.listdir(defect_path):
                    if not is_image_file(img_name):
                        continue

                    img_path = defect_path / img_name

                    records.append(
                        {
                            "image_path": str(img_path),
                            "category": category,
                            "split": split,
                            "defect_type": defect_type,
                            "label": label,
                            "is_normal": label == normal_label,
                        }
                    )

    df = pd.DataFrame.from_records(records)

    if df.empty:
        raise RuntimeError("Dataset index is empty. Check dataset path or config.")

    return df


def main():
    config_path = "configs/data.yaml"
    output_path = Path("data/processed/dataset_index.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    df = index_mvtec_dataset(cfg)

    df.to_csv(output_path, index=False)

    print("Dataset indexing complete.")
    print(f"Saved to: {output_path}")
    print("\nSample:")
    print(df.head())
    print("\nClass distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
