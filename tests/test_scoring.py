import pytest
import torch

from src.scoring import build_scorer

METHODS = ["l2_prototype", "mahalanobis", "knn"]


@pytest.fixture
def embeddings():
    g = torch.Generator().manual_seed(0)
    normal = torch.randn(200, 16, generator=g)
    shifted = torch.randn(50, 16, generator=g) + 4.0
    return normal, shifted


@pytest.mark.parametrize("method", METHODS)
def test_outliers_score_higher(method, embeddings):
    normal, shifted = embeddings
    scorer = build_scorer(method, cov_eps=0.01, knn_k=5).fit(normal[:150])
    s_normal = scorer.score(normal[150:])
    s_shift = scorer.score(shifted)
    assert s_normal.shape == (50,)
    assert s_shift.min() > s_normal.median()


@pytest.mark.parametrize("method", METHODS)
def test_state_roundtrip(method, embeddings):
    normal, shifted = embeddings
    a = build_scorer(method).fit(normal)
    b = build_scorer(method).load_state_dict(a.state_dict())
    assert torch.allclose(a.score(shifted), b.score(shifted))


def test_mahalanobis_handles_fewer_samples_than_dims():
    x = torch.randn(10, 64)
    s = build_scorer("mahalanobis", cov_eps=0.01).fit(x).score(torch.randn(5, 64))
    assert torch.isfinite(s).all()


def test_unknown_method():
    with pytest.raises(ValueError):
        build_scorer("nope")
