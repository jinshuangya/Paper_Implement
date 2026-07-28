"""Restricted separation of Lagrangian cuts (Algorithm 1 of the paper).

A Lagrangian cut has the form (eq. 13)

    pi @ x + pi0 * theta_s >= Q*_s(pi, pi0),

and separating a candidate (x_hat, theta_hat) means solving (eq. 17)

    max_{(pi, pi0) in Pi_s}  Q*_s(pi, pi0) - pi @ x_hat - pi0 * theta_hat.

``Pi_s`` is a *restriction* of the coefficient cone R^n x R_+.  The paper's
choices, all implemented here, are:

* ``exact``  : Pi_s = { alpha*pi0 + ||pi||_1 <= 1, pi0 >= 0 }         (full space)
* ``rstr1``  : pi = sum_k beta_k pi^k,  alpha*pi0 + ||pi||_1 <= 1      (eq. 19)
* ``rstr2``  : pi = sum_k beta_k pi^k,  alpha*pi0 + ||beta||_1 <= 1    (eq. 20)

The function Q*_s is replaced during separation by an upper-bounding
cutting-plane model  Qbar*_s(pi, pi0) = min_{(z, theta^z) in Ehat_s} pi@z +
pi0*theta^z (eq. 18); every evaluation of the true Q*_s adds the returned point
to the pool ``Ehat_s``, tightening the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import highspy
import numpy as np

from .oracle import eval_qstar, eval_recourse
from .sslp import SSLPInstance


# ---------------------------------------------------------------------------
# Pool Ehat_s : the finite set of (z, theta^z) points defining Qbar*_s
# ---------------------------------------------------------------------------
@dataclass
class ScenarioPool:
    """Points (z, theta_z) with z a feasible first-stage vector and theta_z a
    feasible second-stage cost, defining the upper model Qbar*_s (eq. 18)."""

    z: list[np.ndarray] = field(default_factory=list)
    theta: list[float] = field(default_factory=list)

    def add(self, z: np.ndarray, theta_z: float) -> None:
        # Zero out sub-threshold floating-point dirt: HiGHS rejects nonzero
        # matrix coefficients below its small-value tolerance (~1e-9).
        z = np.asarray(z, dtype=float).copy()
        z[np.abs(z) < 1e-9] = 0.0
        tz = float(theta_z)
        if abs(tz) < 1e-9:
            tz = 0.0
        self.z.append(z)
        self.theta.append(tz)

    def __len__(self) -> int:
        return len(self.z)


@dataclass
class LagrangianCut:
    """Cut  pi @ x + pi0 * theta_s >= rhs   valid for E_s (eq. 13)."""

    pi: np.ndarray
    pi0: float
    rhs: float  # = Q*_s(pi, pi0)

    def violation(self, x: np.ndarray, theta_s: float) -> float:
        return self.rhs - float(self.pi @ x) - self.pi0 * float(theta_s)


@dataclass
class PiSpec:
    """Specification of the restricted coefficient set Pi_s."""

    mode: str  # "exact" | "rstr1" | "rstr2"
    alpha: float = 1.0
    basis: np.ndarray | None = None  # (K, m) basis vectors pi^k for restricted modes


# ---------------------------------------------------------------------------
# Restricted separation master (eq. 17 with Qbar* in place of Q*)
# ---------------------------------------------------------------------------
_COEF_TOL = 1e-9  # HiGHS rejects nonzero matrix coefficients below ~this


def _lin(coeffs, vars_):
    """Build a HiGHS linear expression sum_i coeffs[i]*vars_[i], dropping any
    coefficient with |.| < _COEF_TOL (HiGHS rejects them).  Returns ``None`` for
    an empty expression."""
    expr = None
    for c, v in zip(coeffs, vars_):
        c = float(c)
        if abs(c) >= _COEF_TOL:
            term = c * v
            expr = term if expr is None else expr + term
    return expr


def _solve_restricted_master(
    pool: ScenarioPool,
    x_hat: np.ndarray,
    theta_hat: float,
    spec: PiSpec,
    m: int,
) -> tuple[float, np.ndarray, float]:
    """Return (obj, pi, pi0) maximising  Qbar*(pi,pi0) - pi@x_hat - pi0*theta_hat
    over Pi_s.  Qbar* is modelled by an epigraph variable ``t`` with
    ``t <= pi@z + pi0*theta_z`` for every pooled point.

    Restricted modes are built directly in the beta (coefficient) space with the
    reduced coefficients computed in numpy and filtered, so an orthonormal SVD
    basis (whose entries can be below the solver's coefficient tolerance) never
    reaches HiGHS as a sub-threshold nonzero.
    """
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    INF = highspy.kHighsInf

    pi0 = h.addVariable(lb=0.0, ub=INF)
    t = h.addVariable(lb=-INF, ub=INF)

    if spec.mode == "exact":
        # decision vars are pi directly; pi = c-space, map to pi is identity.
        cvars = h.addVariables(m, lb=-INF, ub=INF)
        B = np.eye(m)  # pi = B^T @ c
        ap = h.addVariables(m, lb=0.0, ub=INF)  # |pi_j|
        for j in range(m):
            h.addConstr(ap[j] >= cvars[j])
            h.addConstr(ap[j] >= -cvars[j])
        norm_terms = _lin(np.full(m, 1.0), [ap[j] for j in range(m)])
        h.addConstr(spec.alpha * pi0 + norm_terms <= 1.0)
    else:
        assert spec.basis is not None, "restricted modes need a basis"
        B = np.asarray(spec.basis, dtype=float)  # (K, m):  pi = B^T @ beta
        K = B.shape[0]
        cvars = h.addVariables(K, lb=-INF, ub=INF)  # beta
        if spec.mode == "rstr1":
            ap = h.addVariables(m, lb=0.0, ub=INF)  # |pi_j|,  pi_j = B[:,j]·beta
            for j in range(m):
                pij = _lin(B[:, j], [cvars[k] for k in range(K)])
                if pij is not None:
                    h.addConstr(ap[j] >= pij)
                    h.addConstr(ap[j] >= -pij)
            norm_terms = _lin(np.full(m, 1.0), [ap[j] for j in range(m)])
            h.addConstr(spec.alpha * pi0 + norm_terms <= 1.0)
        elif spec.mode == "rstr2":
            ab = h.addVariables(K, lb=0.0, ub=INF)  # |beta_k|
            for k in range(K):
                h.addConstr(ab[k] >= cvars[k])
                h.addConstr(ab[k] >= -cvars[k])
            norm_terms = _lin(np.full(K, 1.0), [ab[k] for k in range(K)])
            h.addConstr(spec.alpha * pi0 + norm_terms <= 1.0)
        else:
            raise ValueError(f"unknown mode {spec.mode}")

    nc = B.shape[0]
    clist = [cvars[k] for k in range(nc)]

    # epigraph of Qbar*: t <= pi@z + tz*pi0 = (B@z)·c + tz*pi0
    for z, tz in zip(pool.z, pool.theta):
        rhs = _lin(B @ z, clist)
        if abs(tz) >= _COEF_TOL:
            term = float(tz) * pi0
            rhs = term if rhs is None else rhs + term
        h.addConstr(t <= (0.0 if rhs is None else rhs))

    # objective: t - pi@x_hat - theta_hat*pi0 = t - (B@x_hat)·c - theta_hat*pi0
    obj = t
    obj_c = _lin(-(B @ x_hat), clist)
    if obj_c is not None:
        obj = obj + obj_c
    if abs(theta_hat) >= _COEF_TOL:
        obj = obj - float(theta_hat) * pi0
    h.maximize(obj)

    h.run()
    obj_val = float(h.getObjectiveValue())
    col = np.array(h.getSolution().col_value, dtype=float)
    c_val = col[[cvars[k].index for k in range(nc)]]
    pi_val = B.T @ c_val
    pi0_val = float(col[pi0.index])
    return obj_val, pi_val, pi0_val


# ---------------------------------------------------------------------------
# Algorithm 1 : solve the restricted separation problem (17)
# ---------------------------------------------------------------------------
@dataclass
class SeparationResult:
    cut: LagrangianCut | None  # best violated cut found (None if none)
    best_violation: float  # LB at termination
    ub: float  # UB at termination
    n_oracle_calls: int  # number of Q*_s (MIP) evaluations used


def separate(
    inst: SSLPInstance,
    s: int,
    x_hat: np.ndarray,
    theta_hat: float,
    spec: PiSpec,
    pool: ScenarioPool,
    delta: float = 0.5,
    max_iter: int = 100,
    pi0_small: float = 1e-4,
) -> SeparationResult:
    """Run Algorithm 1: iterate master (restricted, uses Qbar*) and oracle
    (true Q*_s via MIP), tightening the pool, until the relative gap is below
    ``delta`` or no violated cut can exist (UB <= 0)."""
    m = inst.m
    ub = np.inf
    lb = -np.inf
    best: LagrangianCut | None = None
    calls = 0

    for _ in range(max_iter):
        ub, pi, pi0 = _solve_restricted_master(pool, x_hat, theta_hat, spec, m)
        if ub <= 1e-9:
            break

        # Evaluate the true Q*_s(pi, pi0) with the single-scenario MIP (12).
        r = eval_qstar(inst, s, pi, pi0)
        calls += 1
        z_pt, theta_pt = r.x, r.theta
        # Small-pi0 safeguard (paper, Section 5.1): the second-stage cost from
        # the MIP can be far from Q_s(x*); recompute the tightest feasible cost.
        if pi0 < pi0_small:
            theta_pt = eval_recourse(inst, s, z_pt)
        pool.add(z_pt, theta_pt)
        qstar = float(pi @ z_pt + pi0 * theta_pt)

        viol = qstar - float(pi @ x_hat) - pi0 * float(theta_hat)
        if viol > lb:
            lb = viol
            best = LagrangianCut(pi=pi.copy(), pi0=pi0, rhs=qstar)

        if ub - lb < delta * ub:  # relative-tolerance stop
            break

    return SeparationResult(cut=best, best_violation=lb, ub=ub, n_oracle_calls=calls)


def init_pool(inst: SSLPInstance, s: int) -> ScenarioPool:
    """Initialise Ehat_s with the perfect-information point (z_s, q_s@y_s) to
    keep Qbar*_s bounded (paper, Section 4.2)."""
    pool = ScenarioPool()
    r = eval_qstar(inst, s, inst.c, 1.0)
    pool.add(r.x, r.theta)
    return pool
