#!/usr/bin/env python
"""Shared pyBANKSY-default spatial-kernel construction."""


from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class BanksyWeights:
    """BANKSY neighborhood-mean and first-order AGF kernels."""

    m0: sp.csr_matrix
    m1: sp.csr_matrix

    @property
    def edge_counts(self) -> dict[str, int]:
        return {"m0": int(self.m0.nnz), "m1": int(self.m1.nnz)}


def generate_banksy_default_weights(
    coordinates: np.ndarray,
    *,
    num_neighbors: int,
    decay_type: str = "scaled_gaussian",
) -> BanksyWeights:
    """Generate pyBANKSY's default max_m=1 pair of spatial kernels."""
    try:
        from banksy.main import generate_spatial_weights_fixed_nbrs
    except ImportError as exc:
        raise RuntimeError("pybanksy is required; run with uv --with pybanksy") from exc

    coordinates = np.asarray(coordinates, dtype=float)
    kernels = []
    for order in (0, 1):
        weights, _, _ = generate_spatial_weights_fixed_nbrs(
            coordinates,
            m=order,
            num_neighbours=num_neighbors,
            decay_type=decay_type,
            verbose=False,
        )
        kernels.append(weights.tocsr())
    return BanksyWeights(m0=kernels[0], m1=kernels[1])


def restrict_banksy_weights_to_labels(
    weights: BanksyWeights,
    labels: np.ndarray,
) -> BanksyWeights:
    """Restrict both BANKSY kernels to same-label neighbors."""
    labels = np.asarray(labels).astype(str)

    def restrict(matrix: sp.csr_matrix, *, order: int) -> sp.csr_matrix:
        coo = matrix.tocoo()
        keep = labels[coo.row] == labels[coo.col]
        filtered = sp.csr_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=matrix.shape,
            dtype=matrix.dtype,
        )
        row_scale = np.asarray(np.abs(filtered).sum(axis=1)).ravel()
        nonzero = row_scale > 0
        if nonzero.any():
            inverse = np.zeros_like(row_scale, dtype=np.float64)
            inverse[nonzero] = 1.0 / row_scale[nonzero]
            filtered = sp.diags(inverse) @ filtered
        if order == 0 and (~nonzero).any():
            isolated = np.flatnonzero(~nonzero)
            filtered = filtered.tolil()
            filtered[isolated, isolated] = 1.0
            filtered = filtered.tocsr()
        return filtered

    return BanksyWeights(
        m0=restrict(weights.m0, order=0),
        m1=restrict(weights.m1, order=1),
    )


def _dense(matrix: np.ndarray | sp.spmatrix) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def banksy_agf(features: np.ndarray | sp.spmatrix, weights: sp.csr_matrix) -> np.ndarray:
    """Apply pyBANKSY's centered first-order azimuthal Gabor filter."""
    features = _dense(features)
    weighted = np.asarray(weights @ features)

    support = weights.copy()
    support.data = np.ones_like(support.data, dtype=np.float64)
    counts = np.asarray(support.sum(axis=1)).ravel()
    means = np.zeros(features.shape, dtype=np.result_type(features.dtype, np.float64))
    nonzero = counts > 0
    if nonzero.any():
        means[nonzero] = np.asarray((support @ features)[nonzero]) / counts[nonzero, None]

    weight_sums = np.asarray(weights.sum(axis=1)).ravel()
    centered = weighted - weight_sums[:, None] * means
    return np.abs(centered)


def zscore_columns(matrix: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Match pyBANKSY's gene-wise z-score operation."""
    matrix = _dense(matrix)
    mean = matrix.mean(axis=0)
    variance = np.square(matrix).mean(axis=0) - np.square(mean)
    variance = np.maximum(variance, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (matrix - mean) / np.sqrt(variance)
    return np.nan_to_num(result).astype(np.float32, copy=False)


def banksy_default_matrix(
    features: np.ndarray | sp.spmatrix,
    weights: BanksyWeights,
    *,
    lambda_value: float = 0.2,
) -> np.ndarray:
    """Construct the default self, m=0, and m=1 BANKSY matrix."""
    if not 0 <= lambda_value <= 1:
        raise ValueError("lambda_value must lie in [0, 1]")

    m0 = weights.m0 @ features
    m1 = banksy_agf(features, weights.m1)
    scale_squared = np.array(
        [1.0 - lambda_value, 2.0 * lambda_value / 3.0, lambda_value / 3.0],
        dtype=np.float32,
    )
    branches = (features, m0, m1)
    return np.concatenate(
        [
            np.sqrt(scale) * zscore_columns(branch)
            for scale, branch in zip(scale_squared, branches)
        ],
        axis=1,
    ).astype(np.float32, copy=False)
