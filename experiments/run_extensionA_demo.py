"""Extension A demonstration: converged bound gap on a rank-collapsed instance.

On the decoupled loose-link/tight-capacity SSLP (where the Benders coefficient
span collapses, see run_basis_rank.py), run every restricted method to *full*
convergence (delta=0) and compare the bound each one can attain.

The point is the converged bound, not speed: a Benders-basis method that stops
with ``stop_reason == "no_cut"`` below the Lagrangian-dual bound has hit a
genuine ceiling -- its restricted cut family cannot express the missing
directions -- whereas the integer-vertex-enriched adaptive subspace (adaptiveA)
reaches further.  ``exact`` (full coefficient cone) is the z_D reference; it is
slow, so allow it plenty of time or pass --no-exact and use adaptiveA as the
reference.

Usage:
    PYTHONPATH=src python experiments/run_extensionA_demo.py \
        --m 6 --n 12 --S 8 --k 2 --capacity-scale 0.5 --time-limit 300
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lagcuts.algo2 import Config, run_algorithm2
from lagcuts.oracle import benders_cut
from lagcuts.reference import solve_extensive_form
from lagcuts.sslp import generate_sslp

COLORS = {"rstr1": "#1f77b4", "rstr2": "#2ca02c", "rstrmip": "#9467bd",
          "adaptiveA": "#ff7f0e", "exact": "#d62728"}


def benders_rank(inst, s=0, samples=60, tol=1e-6):
    rng = np.random.default_rng(0)
    rows = [benders_cut(inst, s, rng.random(inst.m))[0].pi for _ in range(samples)]
    sv = np.linalg.svd(np.vstack(rows), compute_uv=False)
    return int(np.sum(sv > tol * sv[0])) if sv[0] > 0 else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--capacity-scale", type=float, default=0.5)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    inst = generate_sslp(args.m, args.n, args.S, k=args.k,
                         decoupled=True, capacity_scale=args.capacity_scale)
    os.makedirs(args.outdir, exist_ok=True)
    zIP = solve_extensive_form(inst)
    rk = benders_rank(inst)
    print(f"{inst.name}: z_IP={zIP:.4f}  Benders span rank={rk}/{inst.m}")

    methods = ["rstr1", "rstr2", "rstrmip", "adaptiveA"]
    if not args.no_exact:
        methods = ["exact"] + methods

    results = {}
    for method in methods:
        log = run_algorithm2(
            inst, Config(method=method, K=inst.m, delta=0.0, time_limit=args.time_limit)
        )
        results[method] = log
        print(f"  {method:10s} final_LB={log.final_bound:9.4f}  "
              f"stop={log.stop_reason:11s} oracle={log.oracle_calls}")

    # z_D reference: exact if available and converged, else the strongest bound.
    if "exact" in results and results["exact"].stop_reason == "no_cut":
        zD = results["exact"].final_bound
    else:
        zD = max(results[m].final_bound for m in methods)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    labels, gaps, colors = [], [], []
    for method in methods:
        labels.append(method)
        gaps.append(zD - results[method].final_bound)  # >=0, how far below z_D
        colors.append(COLORS[method])
    bars = ax.bar(labels, gaps, color=colors)
    for b, method in zip(bars, methods):
        lg = results[method]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{lg.final_bound:.2f}\n({lg.stop_reason})",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("converged bound gap below z_D  (0 = reaches z_D)")
    ax.set_title(f"Extension A: converged bound on a rank-collapsed instance\n"
                 f"{inst.name}, Benders span rank {rk}/{inst.m}")
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(args.outdir, f"extA_demo_{inst.name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nz_D reference = {zD:.4f}")
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    main()
