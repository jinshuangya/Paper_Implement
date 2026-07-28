"""Extension A: adaptive low-dimensional coefficient subspaces.

The paper's restricted separation fixes the basis {pi^k} to the last K Benders
cut coefficients (rstr1/rstr2) or picks a subset of *all* Benders coefficients
by MIP (rstrmip).  Either way the search directions live inside the span of
Benders LP tangents, which can cap the attainable bound (rstr1's plateau).

This module treats the separation basis as an **online-maintained generating
set**: it collects candidate coefficient directions and returns the top-K
principal directions (truncated SVD) as an orthonormal basis for restricted
separation (eq. 17 / 20).

Candidate directions, both available for free during Algorithm 2:

* Benders cut coefficients (LP tangents of Q_s), and
* the integer first-stage vertices x* returned by the Q*_s oracle (the pooled
  points): supergradients of Q*_s are (x*, theta), so these vertices of
  conv(K_s) carry directions that the Benders tangents alone may miss.

Each direction is L2-normalised before the SVD so that 0/1 vertices and
Benders coefficients (different natural scales) contribute on equal footing.
The singular spectrum is exposed for the "is the optimal-coefficient sequence
really low-rank?" diagnostic that motivates axis A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptiveBasis:
    basis: np.ndarray  # (k, m) orthonormal rows
    singular_values: np.ndarray  # full spectrum of the candidate matrix
    n_candidates: int


def adaptive_basis(
    directions: list[np.ndarray], K: int, energy: float | None = None
) -> AdaptiveBasis | None:
    """Top-K principal directions of the (row-normalised) candidate matrix.

    ``directions`` is a list of (m,) coefficient vectors.  Zero vectors are
    dropped.  If ``energy`` in (0, 1] is given, the rank is instead the smallest
    number of singular vectors capturing that fraction of squared singular
    energy, still capped at ``K`` (an adaptive, spectrum-driven basis size).
    Returns ``None`` when there is no usable direction.
    """
    rows = []
    for v in directions:
        v = np.asarray(v, dtype=float)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-9:
            rows.append(v / nrm)
    if not rows:
        return None

    M = np.vstack(rows)  # (N, m)
    # Right singular vectors Vt are an orthonormal basis of row space, ordered
    # by singular value (dominant coefficient patterns first).
    _, S, Vt = np.linalg.svd(M, full_matrices=False)

    k = min(K, Vt.shape[0])
    if energy is not None and S.size:
        cume = np.cumsum(S**2) / np.sum(S**2)
        k_energy = int(np.searchsorted(cume, energy) + 1)
        k = min(K, max(1, k_energy), Vt.shape[0])

    # keep only directions with non-negligible singular value
    keep = min(k, int(np.sum(S > 1e-9 * (S[0] if S[0] > 0 else 1.0))))
    keep = max(1, keep)
    return AdaptiveBasis(basis=Vt[:keep], singular_values=S, n_candidates=M.shape[0])
