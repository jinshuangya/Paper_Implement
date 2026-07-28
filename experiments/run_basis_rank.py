"""Diagnostic for extension A: does the Benders coefficient span collapse?

The restricted-separation basis of rstr1/rstr2/rstrmip is built entirely from
Benders (LP-dual) coefficients.  Extension A's premise is that this span can be
rank-deficient -- and that the integer vertices x* returned by the Q*_s oracle
supply the missing directions.

This script sweeps the ``capacity_scale`` big-M knob of SSLP and, for one
scenario, measures the effective rank of

  * the set of Benders cut coefficients sampled at random first-stage points, and
  * the set of integer vertices x* sampled at random (pi, pi0) directions,

then plots both against capacity_scale.  As the LP relaxation loosens, the
Benders span collapses while the integer-vertex span stays full -- the
structural signal that motivates an integer-aware adaptive subspace.

Usage:
    PYTHONPATH=src python experiments/run_basis_rank.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lagcuts.oracle import benders_cut, eval_qstar
from lagcuts.sslp import generate_sslp


def effective_rank(vectors: list[np.ndarray], tol: float = 1e-6) -> int:
    if not vectors:
        return 0
    M = np.vstack(vectors)
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * s[0])) if s[0] > 0 else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--S", type=int, default=30)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--samples", type=int, default=80)
    ap.add_argument("--scenario", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    scales = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    rng = np.random.default_rng(0)
    os.makedirs(args.outdir, exist_ok=True)

    bend_rank, vert_rank, lp_looseness = [], [], []
    s = args.scenario
    for cs in scales:
        inst = generate_sslp(args.m, args.n, args.S, args.k, capacity_scale=cs)
        bend, verts = [], []
        for _ in range(args.samples):
            x = rng.random(inst.m)  # fractional first-stage point
            cut, _ = benders_cut(inst, s, x)
            bend.append(cut.pi)
            pi = rng.standard_normal(inst.m)
            pi0 = abs(rng.standard_normal())
            verts.append(eval_qstar(inst, s, pi, pi0).x)
        rb, rv = effective_rank(bend), effective_rank(verts)
        bend_rank.append(rb)
        vert_rank.append(rv)
        lp_looseness.append(cs)
        print(f"capacity_scale={cs:5g}: Benders rank={rb:2d}/{inst.m}  "
              f"vertex rank={rv:2d}/{inst.m}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(scales, bend_rank, "o-", color="#1f77b4", lw=2,
            label="Benders coefficient span")
    ax.plot(scales, vert_rank, "s-", color="#ff7f0e", lw=2,
            label="integer-vertex span (x*)")
    ax.axhline(args.m, ls=":", color="k", lw=1, label=f"full rank (m={args.m})")
    ax.set_xlabel("capacity_scale (big-M looseness of the LP relaxation)")
    ax.set_ylabel("effective rank of the direction set")
    ax.set_title("Extension A premise: Benders span collapses, vertex span does not")
    ax.set_ylim(0, args.m + 0.5)
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    out = os.path.join(args.outdir, "basis_rank_collapse.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nsaved plot -> {out}")


if __name__ == "__main__":
    main()
