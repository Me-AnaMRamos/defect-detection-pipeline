import copy

import pytest

from src.config import load_config, load_yaml, parse_config

CONFIG = "configs/train.yaml"


def test_repo_config_parses():
    cfg = load_config(CONFIG)
    assert cfg.evaluation.target_fpr == 0.05
    assert set(cfg.scoring.methods) <= {"l2_prototype", "mahalanobis", "knn"}


def test_missing_key_is_reported():
    raw = load_yaml(CONFIG)
    del raw["evaluation"]
    with pytest.raises(ValueError, match="evaluation"):
        parse_config(raw)


def test_unknown_method_rejected():
    raw = copy.deepcopy(load_yaml(CONFIG))
    raw["scoring"]["methods"] = ["l2_prototype", "magic"]
    with pytest.raises(ValueError, match="magic"):
        parse_config(raw)
