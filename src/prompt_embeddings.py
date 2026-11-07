"""Prompt embeddings for measured-outcome replay, and the kNN router baseline.

Published routers condition on the prompt text: RouterBench's kNN and MLP
routers, RouteLLM and CARROT all read a sentence embedding.  The controllers
here get the same information in a form their linear predictors and the
zeroth-order optimizers can carry: the embedding reduced by PCA to a few
coordinates and scaled to norm at most one.

LLMRouterBench reports that the choice of embedding model has little effect on
routing quality, so one small, widely used encoder is pinned.  Everything
fitted -- the PCA basis, the scale and the kNN index -- uses the train split
only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import real_benchmarks


ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
# Pinned commit of the encoder weights (head observed on 2026-11-05).
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER_MAX_TOKENS = 256
KNN_NEIGHBOURS = 10
# RouterBench's MLP router, fixed before any run: one hidden layer, early
# stopping on a tenth of the train split, fixed seed.  Nothing is tuned.
MLP_ROUTER = dict(hidden_layer_sizes=(256,), alpha=1e-3, learning_rate_init=1e-3, max_iter=300,
                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=0)


def _cache_path(benchmark: str, root: Path) -> Path:
    return root / "processed" / f"{benchmark}_emb_{ENCODER.rsplit('/', 1)[-1]}.parquet"


def _mlp_cache_path(benchmark: str, root: Path) -> Path:
    return root / "processed" / f"{benchmark}_mlp_quality_{ENCODER.rsplit('/', 1)[-1]}.parquet"


def mlp_quality(benchmark: str, root: Path = real_benchmarks.EXTERNAL) -> pd.DataFrame:
    """Every model's quality on every prompt as predicted by a trained MLP router.

    The input is the full sentence embedding, the output one quality estimate
    per model; the network is fitted on train prompts only.  Cached, so every
    run and every worker process reads the same numbers.  Rows are item ids,
    columns model names.
    """
    path = _mlp_cache_path(benchmark, root)
    if path.exists():
        return pd.read_parquet(path)
    from sklearn.neural_network import MLPRegressor
    items, outcomes = real_benchmarks.load(benchmark, root)
    vectors = encode(benchmark, root).loc[items["item_id"]].to_numpy(np.float64)
    wide = outcomes.pivot(index="item_id", columns="model", values="quality")
    models = sorted(wide.columns)
    quality = wide.loc[items["item_id"], models].to_numpy(float)
    train = (items["split"] == "train").to_numpy()
    network = MLPRegressor(**MLP_ROUTER).fit(vectors[train], quality[train])
    frame = pd.DataFrame(np.clip(network.predict(vectors), 0.0, 1.0), index=items["item_id"],
                         columns=models)
    frame.to_parquet(path)
    return frame


def encode(benchmark: str, root: Path = real_benchmarks.EXTERNAL, batch_size: int = 128) -> pd.DataFrame:
    """Unit-norm sentence embeddings of every prompt, cached per benchmark."""
    path = _cache_path(benchmark, root)
    if path.exists():
        return pd.read_parquet(path)
    from sentence_transformers import SentenceTransformer
    items, _ = real_benchmarks.load(benchmark, root)
    model = SentenceTransformer(ENCODER, device="cpu", revision=ENCODER_REVISION)
    model.max_seq_length = ENCODER_MAX_TOKENS
    vectors = model.encode(items["prompt"].astype(str).tolist(), batch_size=batch_size,
                           normalize_embeddings=True, show_progress_bar=True,
                           convert_to_numpy=True)
    frame = pd.DataFrame(vectors.astype(np.float32), index=items["item_id"])
    frame.columns = [f"e{i}" for i in range(frame.shape[1])]
    frame.to_parquet(path)
    return frame


class PromptFeatures:
    """Reduced embeddings and kNN quality estimates for one benchmark."""

    def __init__(self, benchmark: str, dim: int, root: Path = real_benchmarks.EXTERNAL,
                 neighbours: int = KNN_NEIGHBOURS):
        items, outcomes = real_benchmarks.load(benchmark, root)
        self.benchmark, self.root = benchmark, root
        self._mlp: np.ndarray | None = None
        self._vectors: np.ndarray | None = None
        self.item_ids = list(items["item_id"])
        self.index = {item: i for i, item in enumerate(self.item_ids)}
        self.dim = int(dim)
        wide = outcomes.pivot(index="item_id", columns="model", values="quality")
        self.models = sorted(wide.columns)
        # The reduction and the kNN estimates are cached: every worker process
        # built them from the full embedding matrix, and the transient memory
        # of doing so in parallel was the largest allocation of a run.
        cache = root / "processed" / (f"{benchmark}_features_{ENCODER.rsplit('/', 1)[-1]}"
                                      f"_d{self.dim}_k{neighbours}.npz")
        if cache.exists():
            with np.load(cache) as stored:
                self.reduced, self.knn_quality = stored["reduced"], stored["knn"]
            return
        train = (items["split"] == "train").to_numpy()
        self.reduced = self._reduce(train) if dim else np.zeros((len(items), 0))
        quality = wide.loc[self.item_ids, self.models].to_numpy(float)
        self.knn_quality = self._knn(train, quality, neighbours)
        np.savez(cache, reduced=self.reduced, knn=self.knn_quality)

    @property
    def vectors(self) -> np.ndarray:
        """The full unit-norm embedding matrix, loaded on first use; runs never need it."""
        if self._vectors is None:
            items, _ = real_benchmarks.load(self.benchmark, self.root)
            self._vectors = encode(self.benchmark, self.root).loc[items["item_id"]].to_numpy(np.float64)
        return self._vectors

    def _reduce(self, train: np.ndarray) -> np.ndarray:
        """PCA fitted on the train split, scaled so train norms are at most one."""
        mean = self.vectors[train].mean(axis=0)
        centred = self.vectors[train] - mean
        _, _, components = np.linalg.svd(centred, full_matrices=False)
        basis = components[: self.dim].T
        projected = (self.vectors - mean) @ basis
        scale = float(np.linalg.norm(projected[train], axis=1).max())
        projected /= max(scale, 1e-12)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        # Validation and test prompts may fall slightly outside the train
        # ball; clipping keeps the ||x|| bound the theory constants rely on.
        return projected / np.maximum(norms, 1.0)

    def _knn(self, train: np.ndarray, quality: np.ndarray, k: int) -> np.ndarray:
        """Mean quality of each model over the k nearest train prompts.

        RouterBench's kNN router.  A train prompt never counts itself as its
        own neighbour, so train estimates are leave-one-out.
        """
        train_rows = np.flatnonzero(train)
        position = np.full(len(self.vectors), -1)
        position[train_rows] = np.arange(len(train_rows))
        bank = self.vectors[train_rows]
        estimates = np.empty((len(self.vectors), quality.shape[1]))
        for start in range(0, len(self.vectors), 1024):
            block = self.vectors[start:start + 1024]
            similarity = block @ bank.T
            own = position[start:start + len(block)]
            hit = np.flatnonzero(own >= 0)
            similarity[hit, own[hit]] = -np.inf
            nearest = np.argpartition(-similarity, k, axis=1)[:, :k]
            estimates[start:start + len(block)] = quality[train_rows[nearest]].mean(axis=1)
        return estimates

    def embedding(self, item_id: str) -> tuple[float, ...]:
        return tuple(float(v) for v in self.reduced[self.index[item_id]])

    def knn(self, item_id: str) -> np.ndarray:
        return self.knn_quality[self.index[item_id]]

    def mlp(self, item_id: str) -> np.ndarray:
        """The MLP router's quality estimates for this prompt, one per model."""
        if self._mlp is None:
            frame = mlp_quality(self.benchmark, self.root)
            self._mlp = frame.loc[self.item_ids, self.models].to_numpy(float)
        return self._mlp[self.index[item_id]]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Encode benchmark prompts")
    parser.add_argument("benchmarks", nargs="+", choices=sorted(real_benchmarks.BUILDERS))
    args = parser.parse_args()
    for name in args.benchmarks:
        frame = encode(name)
        print(name, frame.shape, "MLP router estimates", mlp_quality(name).shape)


if __name__ == "__main__":
    main()
