# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Finite GP representer matching proofs/lean/LandfoldFes/Representer.lean.

The noise-free mean on the observation table is μ = Kα with Kα = y.
These helpers build an IMQ Gram, solve that system, and check that
the table is a stable section of K (Eqs. representer, interp, gram-adj).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "adj_k2",
    "apply_k2",
    "coalesce_sites",
    "det_k2",
    "farthest_indices",
    "imq_gram",
    "solve_representer",
    "training_residual",
]


def det_k2(a, b, c):
    """Lean ``detK2``."""
    return a * c - b * b


def adj_k2(a, b, c, y1, y2):
    """Lean ``adjK2``."""
    return (c * y1 - b * y2, -b * y1 + a * y2)


def apply_k2(a, b, c, alpha1, alpha2):
    """Lean ``applyK2``."""
    return (a * alpha1 + b * alpha2, b * alpha1 + c * alpha2)


def imq_gram(xy: np.ndarray, eps: float) -> np.ndarray:
    """Value IMQ Gram on the observation sites."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        msg = "xy must be (n, 2)"
        raise ValueError(msg)
    if not np.isfinite(eps) or eps <= 0.0:
        msg = "eps must be finite and > 0"
        raise ValueError(msg)
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    return 1.0 / np.sqrt(1.0 + d2 / (eps * eps))


def solve_representer(xy: np.ndarray, z: np.ndarray, eps: float, nugget: float = 1e-12):
    """Solve Kα = z on the table. Raises if the Gram is not a representer."""
    z = np.asarray(z, dtype=float).reshape(-1)
    K = imq_gram(xy, eps)
    if z.shape[0] != K.shape[0]:
        msg = "z length must match xy rows"
        raise ValueError(msg)
    Kn = K + nugget * np.eye(K.shape[0])
    cond = float(np.linalg.cond(Kn))
    if not np.isfinite(cond) or cond > 1e12:
        msg = f"Gram is not a stable representer (cond={cond})"
        raise ValueError(msg)
    alpha = np.linalg.solve(Kn, z)
    return alpha, Kn


def training_residual(
    xy: np.ndarray, z: np.ndarray, eps: float, nugget: float = 1e-12
) -> np.ndarray:
    """Kα − z at the observation sites (Eq. interp residual)."""
    alpha, Kn = solve_representer(xy, z, eps, nugget)
    return Kn @ alpha - np.asarray(z, dtype=float).reshape(-1)


def coalesce_sites(s1, s2, z, g1=None, g2=None, ndigits: int = 9):
    """Mean-merge coincident sites so the Gram has one row per location."""
    s1 = np.asarray(s1, dtype=float).reshape(-1)
    s2 = np.asarray(s2, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    keys = list(zip(np.round(s1, ndigits), np.round(s2, ndigits), strict=True))
    buckets: dict[tuple, list[int]] = {}
    for i, key in enumerate(keys):
        buckets.setdefault(key, []).append(i)
    order = list(buckets.values())
    idx = [b[0] for b in order]
    out_s1 = np.array([float(np.mean(s1[b])) for b in order])
    out_s2 = np.array([float(np.mean(s2[b])) for b in order])
    out_z = np.array([float(np.mean(z[b])) for b in order])
    out_g1 = out_g2 = None
    if g1 is not None and g2 is not None:
        g1 = np.asarray(g1, dtype=float).reshape(-1)
        g2 = np.asarray(g2, dtype=float).reshape(-1)
        out_g1 = np.array([float(np.mean(g1[b])) for b in order])
        out_g2 = np.array([float(np.mean(g2[b])) for b in order])
    return out_s1, out_s2, out_g1, out_g2, out_z, np.asarray(idx)


def farthest_indices(xy: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Gonzalez farthest-point subset of the observation table."""
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    k = min(int(k), n)
    if k < 1:
        msg = "k must be >= 1"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    pick = [int(rng.integers(n))]
    d2 = np.full(n, np.inf)
    for _ in range(1, k):
        last = xy[pick[-1]]
        d2 = np.minimum(d2, np.sum((xy - last) ** 2, axis=1))
        pick.append(int(np.argmax(d2)))
    return np.asarray(pick, dtype=int)
