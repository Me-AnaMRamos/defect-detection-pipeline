from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    image_size: int
    train_split: float
    val_split: float


@dataclass(frozen=True)
class ModelConfig:
    name: str
    pretrained: bool


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    lr: float


@dataclass(frozen=True)
class AppConfig:
    seed: int
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Config must be a YAML mapping at top-level. Got: {type(obj)}")
    return obj


def parse_config(raw: Dict[str, Any]) -> AppConfig:
    paths = raw.get("paths", {})
    data = raw.get("data", {})
    model = raw.get("model", {})
    training = raw.get("training", {})

    return AppConfig(
        seed=int(raw.get("seed", 42)),
        paths=PathsConfig(
            data_dir=Path(paths["data_dir"]),
            output_dir=Path(paths["output_dir"]),
        ),
        data=DataConfig(
            image_size=int(data["image_size"]),
            train_split=float(data["train_split"]),
            val_split=float(data["val_split"]),
        ),
        model=ModelConfig(
            name=str(model["name"]),
            pretrained=bool(model["pretrained"]),
        ),
        training=TrainingConfig(
            batch_size=int(training["batch_size"]),
            epochs=int(training["epochs"]),
            lr=float(training["lr"]),
        ),
    )
