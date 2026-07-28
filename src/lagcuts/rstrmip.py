"""RstrMIP basis selection (eq. 21-28 of the paper).

Given a pool of candidate coefficient vectors V_s = {v_k} (all Benders cut
coefficients generated so far for scenario s), pick a subset of size at most K
that maximises the normalised violation, using the current upper-bounding model
Qbar*_s in place of the intractable Q*_s.  The MIP approximation (eq. 28) is

    max   tau - pi @ x_hat - pi0 * theta_hat
    s.t.  tau <= pi @ z + pi0 * theta_z      for (z, theta_z) in Ehat_s
          pi  = sum_k beta_k v_k
          alpha * pi0 + sum_k |beta_k| <= 1
          -z_k <= beta_k <= z_k              (z_k binary select indicator)
          sum_k z_k <= K
          pi0 >= 0.

The selected vectors {v_k : z_k = 1} form the basis fed to restricted
separation with Pi_s of eq. (20).  Returns ``None`` when the MIP's optimal value
is non-positive (no violated Lagrangian cut can be found -> skip the scenario).
"""

from __future__ import annotations

import highspy
import numpy as np

from .separation import ScenarioPool
from .sslp import SSLPInstance


def select_basis_mip(
    inst: SSLPInstance,
    s: int,
    candidates: np.ndarray,  # (Ncand, m)
    pool: ScenarioPool,
    x_hat: np.ndarray,
    theta_hat: float,
    K: int,
    alpha: float,
    tol: float = 1e-6,
) -> np.ndarray | None:
    m = inst.m
    Nc = candidates.shape[0]
    if Nc == 0 or len(pool) == 0:
        return None

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    INF = highspy.kHighsInf

    pi0 = h.addVariable(lb=0.0, ub=INF)
    tau = h.addVariable(lb=-INF, ub=INF)
    beta = h.addVariables(Nc, lb=-1.0, ub=1.0)
    ab = h.addVariables(Nc, lb=0.0, ub=1.0)  # |beta_k|
    zsel = h.addVariables(Nc, lb=0.0, ub=1.0)
    for v in zsel:
        h.setInteger(v)

    # pi_j = sum_k beta_k * candidates[k, j]
    pi_expr = [sum(float(candidates[k, j]) * beta[k] for k in range(Nc)) for j in range(m)]

    for k in range(Nc):
        h.addConstr(ab[k] >= beta[k])
        h.addConstr(ab[k] >= -beta[k])
        h.addConstr(ab[k] <= zsel[k])  # z_k = 0  =>  beta_k = 0
    h.addConstr(alpha * pi0 + sum(ab[k] for k in range(Nc)) <= 1.0)
    h.addConstr(sum(zsel[k] for k in range(Nc)) <= K)

    # tau <= Qbar*  epigraph
    for z, tz in zip(pool.z, pool.theta):
        h.addConstr(tau <= sum(float(z[j]) * pi_expr[j] for j in range(m)) + float(tz) * pi0)

    h.maximize(tau - sum(float(x_hat[j]) * pi_expr[j] for j in range(m)) - float(theta_hat) * pi0)
    h.run()

    obj = float(h.getObjectiveValue())
    if obj <= tol:
        return None

    col = np.array(h.getSolution().col_value, dtype=float)
    zval = col[[zsel[k].index for k in range(Nc)]]
    chosen = [k for k in range(Nc) if zval[k] > 0.5]
    if not chosen:
        return None
    return candidates[chosen]
