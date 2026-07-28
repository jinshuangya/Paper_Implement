"""Algorithm 2: cut-generation loop at the root node.

Each outer iteration solves the Benders master, first exhausts Benders cuts,
and only when no Benders cut is violated does it generate Lagrangian cuts by
restricted separation (Algorithm 1).  The master objective after every solve is
a valid lower bound on z_IP; we log it against wall-clock time to build the
gap-closed profiles of Section 5.2.

Supported ``method`` values:
    "benders"  -- Benders cuts only (LP-relaxation bound)
    "exact"    -- exact Lagrangian separation (Pi_s = full normalised cone)
    "rstr1"    -- restricted, Pi_s eq. (19), basis = last K Benders coeffs
    "rstr2"    -- restricted, Pi_s eq. (20), basis = last K Benders coeffs
    "rstrmip"  -- restricted, Pi_s eq. (20), basis chosen by the MIP (eq. 28)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .master import BendersMaster
from .oracle import benders_cut
from .separation import PiSpec, ScenarioPool, init_pool, separate
from .rstrmip import select_basis_mip
from .sslp import SSLPInstance


@dataclass
class Config:
    method: str = "rstrmip"
    K: int = 20
    alpha: float = 1.0
    delta: float = 0.5
    time_limit: float = 60.0
    benders_tol: float = 1e-4
    lag_tol: float = 1e-6
    max_outer: int = 100000
    verbose: bool = False


@dataclass
class RunLog:
    times: list[float] = field(default_factory=list)  # wall-clock since start
    bounds: list[float] = field(default_factory=list)  # master LB
    events: list[str] = field(default_factory=list)  # "benders" | "lagrangian"
    oracle_calls: int = 0
    lp_bound: float | None = None  # bound after Benders phase converged
    final_bound: float | None = None

    def record(self, t0: float, lb: float, event: str) -> None:
        self.times.append(time.perf_counter() - t0)
        self.bounds.append(lb)
        self.events.append(event)


def _basis_last_k(history: list[np.ndarray], K: int) -> np.ndarray | None:
    if not history:
        return None
    return np.array(history[-K:], dtype=float)


def run_algorithm2(inst: SSLPInstance, cfg: Config) -> RunLog:
    S = inst.S
    master = BendersMaster(inst, integer=False)
    pools: list[ScenarioPool] = [init_pool(inst, s) for s in range(S)]
    benders_hist: list[list[np.ndarray]] = [[] for _ in range(S)]
    log = RunLog()
    t0 = time.perf_counter()

    # Initial Benders cuts at x = 0 to bound theta_s from below (Section 5.1).
    x0 = np.zeros(inst.m)
    for s in range(S):
        cut, _ = benders_cut(inst, s, x0)
        master.add_cut(s, cut.pi, 1.0, cut.const)
        benders_hist[s].append(cut.pi.copy())

    lp_bound_recorded = False
    last_lb = None

    for _ in range(cfg.max_outer):
        if time.perf_counter() - t0 > cfg.time_limit:
            break

        xval, thval, lb = master.solve()
        last_lb = lb

        # ---- Benders phase: add any violated Benders cut ----
        found = False
        for s in range(S):
            cut, _ = benders_cut(inst, s, xval)
            if cut.value_at(xval) - thval[s] > cfg.benders_tol * (abs(thval[s]) + 1.0):
                master.add_cut(s, cut.pi, 1.0, cut.const)
                benders_hist[s].append(cut.pi.copy())
                found = True
        if found:
            log.record(t0, lb, "benders")
            continue

        # Benders phase converged: record the LP-relaxation bound once.
        if not lp_bound_recorded:
            log.lp_bound = lb
            lp_bound_recorded = True
            if cfg.verbose:
                print(f"[benders converged] LB = {lb:.4f}")

        if cfg.method == "benders":
            break

        # ---- Lagrangian phase: restricted separation per scenario ----
        found = False
        for s in range(S):
            spec = _make_spec(inst, cfg, s, benders_hist, pools, xval, thval)
            if spec is None:
                continue
            res = separate(
                inst, s, xval, thval[s], spec, pools[s],
                delta=cfg.delta,
            )
            log.oracle_calls += res.n_oracle_calls
            if res.cut is not None and res.cut.violation(xval, thval[s]) > cfg.lag_tol:
                c = res.cut
                if master.add_cut(s, c.pi, c.pi0, c.rhs):
                    found = True
        if found:
            xval, thval, lb = master.solve()
            last_lb = lb
            log.record(t0, lb, "lagrangian")
        else:
            break

    log.final_bound = last_lb if last_lb is not None else log.lp_bound
    return log


def _make_spec(
    inst: SSLPInstance,
    cfg: Config,
    s: int,
    benders_hist: list[list[np.ndarray]],
    pools: list[ScenarioPool],
    xval: np.ndarray,
    thval: np.ndarray,
) -> PiSpec | None:
    if cfg.method == "exact":
        return PiSpec(mode="exact", alpha=cfg.alpha)
    if cfg.method == "rstr1":
        basis = _basis_last_k(benders_hist[s], cfg.K)
        return None if basis is None else PiSpec("rstr1", cfg.alpha, basis)
    if cfg.method == "rstr2":
        basis = _basis_last_k(benders_hist[s], cfg.K)
        return None if basis is None else PiSpec("rstr2", cfg.alpha, basis)
    if cfg.method == "rstrmip":
        candidates = benders_hist[s]
        if not candidates:
            return None
        basis = select_basis_mip(
            inst, s, np.array(candidates), pools[s], xval, thval[s], cfg.K, cfg.alpha
        )
        return None if basis is None else PiSpec("rstr2", cfg.alpha, basis)
    raise ValueError(f"unknown method {cfg.method}")
