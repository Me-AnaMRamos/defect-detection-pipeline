"""Anomaly scorers fitted on embeddings of normal images only.

Every scorer follows the same contract: ``fit`` on normal train embeddings,
``score`` returns one value per embedding where higher means more anomalous.
"""

from __future__ import annotations

from typing import Any

import torch


class Scorer:
    name: str = "base"

    def fit(self, normal_embeddings: torch.Tensor) -> Scorer:
        raise NotImplementedError

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, Any]) -> Scorer:
        raise NotImplementedError


class L2PrototypeScorer(Scorer):
    """Euclidean distance to the mean normal embedding."""

    name = "l2_prototype"

    def __init__(self) -> None:
        self.prototype: torch.Tensor | None = None

    def fit(self, normal_embeddings: torch.Tensor) -> L2PrototypeScorer:
        self.prototype = normal_embeddings.double().mean(dim=0)
        return self

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        assert self.prototype is not None, "call fit() first"
        return torch.linalg.vector_norm(embeddings.double() - self.prototype, dim=1)

    def state_dict(self) -> dict[str, Any]:
        return {"prototype": self.prototype}

    def load_state_dict(self, state: dict[str, Any]) -> L2PrototypeScorer:
        self.prototype = state["prototype"]
        return self


class MahalanobisScorer(Scorer):
    """Squared Mahalanobis distance under a Gaussian fitted to normal embeddings.

    With fewer samples than feature dimensions the sample covariance is singular,
    so ``eps * I`` is added before inversion.
    """

    name = "mahalanobis"

    def __init__(self, eps: float = 0.01) -> None:
        self.eps = eps
        self.mu: torch.Tensor | None = None
        self.cov_inv: torch.Tensor | None = None

    def fit(self, normal_embeddings: torch.Tensor) -> MahalanobisScorer:
        x = normal_embeddings.double()
        if x.shape[0] < 2:
            raise ValueError("Mahalanobis needs at least 2 normal samples")
        self.mu = x.mean(dim=0)
        centered = x - self.mu
        cov = centered.T @ centered / (x.shape[0] - 1)
        cov += self.eps * torch.eye(cov.shape[0], dtype=cov.dtype)
        self.cov_inv = torch.linalg.inv(cov)
        return self

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        assert self.mu is not None and self.cov_inv is not None, "call fit() first"
        diff = embeddings.double() - self.mu
        return torch.einsum("bi,ij,bj->b", diff, self.cov_inv, diff)

    def state_dict(self) -> dict[str, Any]:
        return {"eps": self.eps, "mu": self.mu, "cov_inv": self.cov_inv}

    def load_state_dict(self, state: dict[str, Any]) -> MahalanobisScorer:
        self.eps, self.mu, self.cov_inv = state["eps"], state["mu"], state["cov_inv"]
        return self


class KNNScorer(Scorer):
    """Mean Euclidean distance to the k nearest normal train embeddings."""

    name = "knn"

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.bank: torch.Tensor | None = None

    def fit(self, normal_embeddings: torch.Tensor) -> KNNScorer:
        if normal_embeddings.shape[0] < self.k:
            raise ValueError(f"kNN needs at least k={self.k} normal samples")
        self.bank = normal_embeddings.double().clone()
        return self

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        assert self.bank is not None, "call fit() first"
        dists = torch.cdist(embeddings.double(), self.bank)
        return dists.topk(self.k, dim=1, largest=False).values.mean(dim=1)

    def state_dict(self) -> dict[str, Any]:
        return {"k": self.k, "bank": self.bank}

    def load_state_dict(self, state: dict[str, Any]) -> KNNScorer:
        self.k, self.bank = state["k"], state["bank"]
        return self


def build_scorer(method: str, cov_eps: float = 0.01, knn_k: int = 5) -> Scorer:
    if method == "l2_prototype":
        return L2PrototypeScorer()
    if method == "mahalanobis":
        return MahalanobisScorer(eps=cov_eps)
    if method == "knn":
        return KNNScorer(k=knn_k)
    raise ValueError(f"Unknown scoring method '{method}'")
