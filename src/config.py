"""Typed configuration loaded from YAML.

Every value the pipeline uses is declared here, so a missing or misspelt key
fails loudly at load time instead of deep inside a run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_METHODS = ("l2_prototype", "mahalanobis", "knn")


@dataclass(frozen=True)
class PathsConfig:
    dataset_index: Path
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    image_size: int
    categories: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    pretrained: bool


@dataclass(frozen=True)
class ScoringConfig:
    methods: tuple[str, ...]
    cov_eps: float
    knn_k: int


@dataclass(frozen=True)
class EvaluationConfig:
    target_fpr: float


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int
    num_workers: int
    device: str


@dataclass(frozen=True)
class AppConfig:
    seed: int
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    scoring: ScoringConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        return d


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Config must be a YAML mapping at top level, got {type(obj).__name__}")
    return obj


def parse_config(raw: dict[str, Any]) -> AppConfig:
    try:
        paths, data, model = raw["paths"], raw["data"], raw["model"]
        scoring, evaluation = raw["scoring"], raw["evaluation"]
        runtime = raw.get("runtime", {})

        cfg = AppConfig(
            seed=int(raw.get("seed", 42)),
            paths=PathsConfig(
                dataset_index=Path(paths["dataset_index"]),
                output_dir=Path(paths["output_dir"]),
            ),
            data=DataConfig(
                image_size=int(data["image_size"]),
                categories=tuple(str(c) for c in data["categories"]),
            ),
            model=ModelConfig(
                backbone=str(model.get("backbone", "resnet18")),
                pretrained=bool(model.get("pretrained", True)),
            ),
            scoring=ScoringConfig(
                methods=tuple(str(m) for m in scoring["methods"]),
                cov_eps=float(scoring.get("cov_eps", 0.01)),
                knn_k=int(scoring.get("knn_k", 5)),
            ),
            evaluation=EvaluationConfig(target_fpr=float(evaluation["target_fpr"])),
            runtime=RuntimeConfig(
                batch_size=int(runtime.get("batch_size", 32)),
                num_workers=int(runtime.get("num_workers", 2)),
                device=str(runtime.get("device", "auto")),
            ),
        )
    except KeyError as e:
        raise ValueError(f"Missing required config key: {e}") from e

    _validate(cfg)
    return cfg


def _validate(cfg: AppConfig) -> None:
    unknown = set(cfg.scoring.methods) - set(VALID_METHODS)
    if unknown:
        raise ValueError(f"Unknown scoring method(s) {sorted(unknown)}; valid: {VALID_METHODS}")
    if not cfg.data.categories:
        raise ValueError("data.categories must list at least one category")
    if not 0.0 < cfg.evaluation.target_fpr < 1.0:
        raise ValueError("evaluation.target_fpr must be in (0, 1)")
    if cfg.scoring.knn_k < 1:
        raise ValueError("scoring.knn_k must be >= 1")
    if cfg.runtime.device not in ("auto", "cpu", "cuda"):
        raise ValueError("runtime.device must be auto, cpu or cuda")


def load_config(path: str | Path) -> AppConfig:
    return parse_config(load_yaml(path))
