"""Stochastic Server Location Problem (SSLP) instance generation.

Follows the data-generation recipe in the appendix of

    Rui Chen and James Luedtke,
    "On Generating Lagrangian Cuts for Two-Stage Stochastic Integer Programs",
    arXiv:2106.04023v2 (2022).

First stage:  x_j in {0,1}, j = 1..m   (locate a server at site j, cost c_j)
Second stage (scenario s):
    y_ij in {0,1}   client i served at site j
    y0_j >= 0       resource shortage at site j
Objective (minimise):
    sum_j c_j x_j + E_s[ sum_j q0_j y0_j - sum_{i,j} q_ij y_ij ]
subject to, for every scenario s:
    sum_i d_ij y_ij - y0_j <= u x_j          (capacity / linking)
    sum_j y_ij           = h^s_i             (present clients must be served)

The recourse is always feasible (shortage variables absorb any overload), i.e.
the problem has relatively complete recourse and no feasibility cuts are needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SSLPInstance:
    """A single SSLP instance with a finite scenario set."""

    m: int  # number of sites (first-stage binary variables)
    n: int  # number of clients
    S: int  # number of scenarios

    c: np.ndarray  # (m,)      first-stage costs
    q: np.ndarray  # (n, m)    revenue q_ij for serving client i at site j
    d: np.ndarray  # (n, m)    resource demand d_ij  (== q_ij in this generator)
    q0: np.ndarray  # (m,)     unit shortage penalty per site
    u: float  # scalar server capacity
    p: np.ndarray  # (S,)      scenario probabilities
    h: np.ndarray  # (S, n)    client availability (0/1) per scenario

    # Decoupled ("loose-link + tight-capacity") formulation.  When
    # ``link_bigM`` is None the standard coupled constraint
    #     sum_i d_ij y_ij - y0_j <= u x_j
    # is used.  When set, the linking and the capacity are split into
    #     sum_i d_ij y_ij        <= link_bigM * x_j     (loose big-M linking)
    #     sum_i d_ij y_ij - y0_j <= u                   (tight capacity, no x)
    # which loosens the LP relaxation (collapsing the Benders direction) while
    # keeping the integer problem hard (opening a site is all-or-nothing).
    link_bigM: float | None = None

    name: str = "sslp"

    @property
    def num_first_stage(self) -> int:
        return self.m


def generate_sslp(
    m: int,
    n: int,
    S: int,
    k: int = 1,
    capacity_scale: float = 1.0,
    decoupled: bool = False,
) -> SSLPInstance:
    """Generate an SSLP instance following the paper's appendix recipe.

    ``k`` is the instance index; together with ``(m, n, S)`` it seeds the RNG so
    that instances are reproducible.  The paper does not publish its seeds, so
    these instances match the *distribution* and structure of the paper's test
    set rather than the exact numbers.

    ``capacity_scale`` multiplies the server capacity ``u``.  With
    ``capacity_scale = 1`` this is the standard SSLP.  Larger values turn the
    capacity linking into a loose big-M constraint: in the LP relaxation a tiny
    fractional ``x_j`` already supplies ample capacity, so the linking is slack
    and its dual (hence the Benders cut direction) degenerates toward zero.  It
    is the tunable knob for the "beyond Benders" (extension A) experiments.
    """
    # Deterministic, collision-resistant seed from the instance signature.
    seed = (m, n, S, k)
    rng = np.random.default_rng(abs(hash(seed)) % (2**32))

    c = rng.integers(40, 81, size=m).astype(float)  # Uniform{40,...,80}
    # d_ij = q_ij ~ Uniform{0,...,25}
    d = rng.integers(0, 26, size=(n, m)).astype(float)
    q = d.copy()
    q0 = np.full(m, 1000.0)
    u = float(capacity_scale * d.sum() / m)
    p = np.full(S, 1.0 / S)
    h = rng.integers(0, 2, size=(S, n)).astype(float)  # Bernoulli(1/2)

    # Decoupled formulation: a big-M that never binds at an integer solution
    # (x_j = 1 already permits serving every client at site j).
    link_bigM = float(d.sum()) if decoupled else None

    if decoupled:
        tag = "sslpD"
    elif capacity_scale != 1.0:
        tag = f"sslpM{capacity_scale:g}"
    else:
        tag = "sslp"
    return SSLPInstance(
        m=m, n=n, S=S, c=c, q=q, d=d, q0=q0, u=u, p=p, h=h,
        link_bigM=link_bigM,
        name=f"{tag}{k}_{m}_{n}_{S}",
    )
