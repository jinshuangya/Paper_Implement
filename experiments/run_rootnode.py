"""Root-node bound-over-time comparison of the Lagrangian-cut generators.

Runs the reproduced methods on one SSLP instance and plots the lower bound
against wall-clock time (the convergence profiles of the paper's Section 5.2),
plus a summary table of the fraction of the Lagrangian-dual gap closed.

Usage:
    PYTHONPATH=src python experiments/run_rootnode.py [--m M --n N --S S --k K]
                                                      [--delta D --basis K]
                                                      [--time-limit T]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lagcuts.algo2 import Config, run_algorithm2
from lagcuts.reference import solve_extensive_form
from lagcuts.sslp import generate_sslp

METHODS = ["benders", "exact", "rstr1", "rstr2", "rstrmip"]
COLORS = {
    "benders": "#888888",
    "exact": "#d62728",
    "rstr1": "#1f77b4",
    "rstr2": "#2ca02c",
    "rstrmip": "#9467bd",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--S", type=int, default=15)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--basis", type=int, default=20, help="basis size K")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    inst = generate_sslp(m=args.m, n=args.n, S=args.S, k=args.k)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"instance {inst.name}: m={inst.m} n={inst.n} S={inst.S}")
    zIP = solve_extensive_form(inst)
    print(f"z_IP (extensive form) = {zIP:.4f}")

    logs = {}
    for method in METHODS:
        cfg = Config(
            method=method, K=args.basis, alpha=1.0, delta=args.delta,
            time_limit=args.time_limit,
        )
        logs[method] = run_algorithm2(inst, cfg)

    lp = logs["benders"].lp_bound
    best = max(l.final_bound for l in logs.values())  # best bound any method got

    print(f"\nLP bound = {lp:.4f}   best Lagrangian bound = {best:.4f}")
    print(f"{'method':9s} {'final_LB':>12s} {'gap_closed%':>12s} {'oracle':>8s}")
    for method in METHODS:
        lg = logs[method]
        closed = (
            100.0 * (lg.final_bound - lp) / (best - lp) if best - lp > 1e-9 else 0.0
        )
        print(f"{method:9s} {lg.final_bound:12.4f} {closed:12.2f} {lg.oracle_calls:8d}")

    # ---- plot: lower bound vs time, focused on the [LP, z_IP] gap region ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    last_t = 0.0
    for method in METHODS:
        lg = logs[method]
        if not lg.times:
            continue
        # hold the final bound out to each method's last event (flat tail)
        t = list(lg.times) + [lg.times[-1]]
        b = list(lg.bounds) + [lg.final_bound]
        last_t = max(last_t, lg.times[-1])
        ax.plot(t, b, drawstyle="steps-post", label=method,
                color=COLORS[method], lw=2)
    ax.axhline(zIP, ls="--", color="k", lw=1, label="z_IP")
    ax.axhline(lp, ls=":", color="#888888", lw=1, label="LP bound")

    # focus the y-axis on the meaningful band (early iterates are far below)
    span = max(zIP - lp, 1.0)
    ax.set_ylim(lp - 0.08 * span, zIP + 0.15 * span)
    ax.set_xlim(0, last_t * 1.1 if last_t > 0 else 1.0)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("lower bound")
    ax.set_title(f"Root-node bound vs time — {inst.name} (delta={args.delta}, K={args.basis})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    out = os.path.join(args.outdir, f"bound_vs_time_{inst.name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nsaved plot -> {out}")


if __name__ == "__main__":
    main()
