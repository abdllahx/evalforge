from __future__ import annotations

import warnings

import hdbscan
import numpy as np
import umap

# UMAP emits a noisy "n_jobs value 1 overridden to 1 by setting random_state"
# on every call — we set the seed deliberately for reproducibility.
warnings.filterwarnings(
    "ignore", message=".*n_jobs value 1 overridden.*", category=UserWarning
)


def cluster(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    reduce_dim: int = 8,
) -> tuple[np.ndarray, hdbscan.HDBSCAN]:
    """HDBSCAN on UMAP-reduced embeddings.

    Short-text MiniLM vectors are 384-d; HDBSCAN struggles with that
    dimensionality and tends to over-merge. Reducing to ~5–10 dims with UMAP
    first (cosine metric) gives much better separation in practice.
    """
    if len(embeddings) > reduce_dim + 2:
        reducer = umap.UMAP(
            n_neighbors=min(15, max(2, len(embeddings) - 1)),
            n_components=reduce_dim,
            random_state=42,
            metric="cosine",
        )
        reduced = reducer.fit_transform(embeddings)
    else:
        reduced = embeddings
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)
    return labels, clusterer


def project_2d(
    embeddings: np.ndarray,
    *,
    n_neighbors: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """UMAP → 2D for the dashboard scatter plot."""
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(2, len(embeddings) - 1)),
        n_components=2,
        random_state=random_state,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)


def representative_indices(
    embeddings: np.ndarray, member_indices: list[int], k: int = 5
) -> list[int]:
    """Pick k representative members closest to the cluster centroid."""
    if len(member_indices) <= k:
        return member_indices
    members = embeddings[member_indices].astype(np.float64, copy=False)
    centroid = members.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-3:
        return member_indices[:k]
    centroid = centroid / norm
    with np.errstate(over="ignore", invalid="ignore"):
        sims = members @ centroid
    sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)
    top = np.argsort(-sims)[:k]
    return [member_indices[int(i)] for i in top]
