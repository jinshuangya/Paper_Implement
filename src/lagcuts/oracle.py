"""Single-scenario oracles for SSLP.

Everything the cut generators need from a scenario subproblem lives here:

* :func:`eval_recourse`      -- Q_s(x): the exact (MIP) recourse value at a
                                fixed first-stage point x.
* :func:`benders_cut`        -- a Benders optimality cut from the LP relaxation
                                of the recourse: coefficients (pi, const) with
                                ``pi @ x + theta_s >= const`` valid for E_s.
* :func:`eval_qstar`         -- Q*_s(pi, pi0) = min { pi@x + pi0 (q_s@y) :
                                (x, y) in K_s }, the homogeneous value function
                                (12) whose optimal solution yields both the
                                function value and a supergradient (x*, q_s@y*).

Second-stage structure for scenario s (given the availability vector h^s):
    variables  y_ij in {0,1}   (or [0,1] for the LP relaxation / continuous),
               y0_j >= 0       (shortage)
    objective  q_s @ y  =  sum_j q0_j y0_j - sum_{i,j} q_ij y_ij
    (A) capacity:   sum_i d_ij y_ij - y0_j <= u x_j          for each site j
    (B) assignment: sum_j y_ij = h^s_i                       for each client i
"""

from __future__ import annotations

from dataclasses import dataclass

import highspy
import numpy as np

from .sslp import SSLPInstance


def _silent() -> highspy.Highs:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    return h


@dataclass
class BendersCut:
    """Optimality cut  pi @ x + theta_s >= const  valid for E_s."""

    pi: np.ndarray  # (m,)
    const: float

    def value_at(self, x: np.ndarray) -> float:
        """Lower bound on Q_s implied by the cut: const - pi @ x."""
        return self.const - float(self.pi @ x)


@dataclass
class QStarResult:
    value: float  # Q*_s(pi, pi0)
    x: np.ndarray  # (m,) optimal first-stage point (supergradient part)
    theta: float  # q_s @ y* (second-stage cost at optimum; supergradient part)


# ---------------------------------------------------------------------------
# Recourse value Q_s(x)  (x fixed, second stage is an integer program)
# ---------------------------------------------------------------------------
def eval_recourse(inst: SSLPInstance, s: int, x: np.ndarray, relax: bool = False) -> float:
    """Exact recourse value Q_s(x).  With ``relax=True`` the binaries y_ij are
    relaxed to [0, 1] (used only for diagnostics)."""
    h = _silent()
    n, m = inst.n, inst.m
    hs = inst.h[s]

    # y_ij flattened as i*m + j ; then y0_j.
    y = h.addVariables(n * m, lb=0.0, ub=1.0)
    y0 = h.addVariables(m, lb=0.0, ub=highspy.kHighsInf)
    if not relax:
        for v in y:
            h.setInteger(v)

    # objective
    obj = sum(inst.q0[j] * y0[j] for j in range(m))
    obj -= sum(inst.q[i, j] * y[i * m + j] for i in range(n) for j in range(m))
    h.minimize(obj)

    # (A) capacity with x fixed
    for j in range(m):
        h.addConstr(
            sum(inst.d[i, j] * y[i * m + j] for i in range(n)) - y0[j]
            <= inst.u * float(x[j])
        )
    # (B) assignment
    for i in range(n):
        h.addConstr(sum(y[i * m + j] for j in range(m)) == float(hs[i]))

    h.run()
    return float(h.getObjectiveValue())


# ---------------------------------------------------------------------------
# Benders cut from the LP relaxation of the recourse
# ---------------------------------------------------------------------------
def benders_cut(inst: SSLPInstance, s: int, x: np.ndarray) -> tuple[BendersCut, float]:
    """Return (cut, Q_LP) where ``cut`` is a Benders optimality cut for E_s and
    ``Q_LP`` is the LP-relaxation recourse value at x (the cut is tight there)."""
    h = _silent()
    n, m = inst.n, inst.m
    hs = inst.h[s]

    y = h.addVariables(n * m, lb=0.0, ub=1.0)
    y0 = h.addVariables(m, lb=0.0, ub=highspy.kHighsInf)

    obj = sum(inst.q0[j] * y0[j] for j in range(m))
    obj -= sum(inst.q[i, j] * y[i * m + j] for i in range(n) for j in range(m))
    h.minimize(obj)

    cap_rows = []
    for j in range(m):
        cap_rows.append(
            h.addConstr(
                sum(inst.d[i, j] * y[i * m + j] for i in range(n)) - y0[j]
                <= inst.u * float(x[j])
            )
        )
    asg_rows = []
    for i in range(n):
        asg_rows.append(h.addConstr(sum(y[i * m + j] for j in range(m)) == float(hs[i])))

    h.run()
    q_lp = float(h.getObjectiveValue())
    row_dual = np.array(h.getSolution().row_dual, dtype=float)

    # Capacity row j:  sum_i d_ij y_ij - y0_j <= u x_j
    # Its dual mu_j gives sensitivity d(obj)/d(rhs) = d(obj)/d(u x_j).
    # The Benders cut is  theta_s >= Q_LP(x_hat) + sum_j (u mu_j)(x_j - x_hat_j),
    # i.e.  pi @ x + theta_s >= const   with  pi_j = -(u mu_j),
    #       const = Q_LP(x_hat) - sum_j (u mu_j) x_hat_j.
    mu_cap = row_dual[[cap_rows[j].index for j in range(m)]]
    pi = -(inst.u * mu_cap)
    const = q_lp - float(pi @ x)
    return BendersCut(pi=pi, const=const), q_lp


# ---------------------------------------------------------------------------
# Homogeneous value function Q*_s(pi, pi0)  (single-scenario MIP, eq. (12))
# ---------------------------------------------------------------------------
def eval_qstar(
    inst: SSLPInstance, s: int, pi: np.ndarray, pi0: float
) -> QStarResult:
    """Solve  min { pi@x + pi0 (q_s@y) : (x, y) in K_s }  and return the value
    together with the supergradient (x*, q_s@y*)."""
    h = _silent()
    n, m = inst.n, inst.m
    hs = inst.h[s]

    x = h.addVariables(m, lb=0.0, ub=1.0)
    for v in x:
        h.setInteger(v)
    y = h.addVariables(n * m, lb=0.0, ub=1.0)
    for v in y:
        h.setInteger(v)
    y0 = h.addVariables(m, lb=0.0, ub=highspy.kHighsInf)

    second_stage_cost = sum(inst.q0[j] * y0[j] for j in range(m)) - sum(
        inst.q[i, j] * y[i * m + j] for i in range(n) for j in range(m)
    )
    obj = sum(float(pi[j]) * x[j] for j in range(m)) + float(pi0) * second_stage_cost
    h.minimize(obj)

    # (A) capacity, x now a variable:  sum_i d_ij y_ij - y0_j <= u x_j
    for j in range(m):
        h.addConstr(
            sum(inst.d[i, j] * y[i * m + j] for i in range(n)) - y0[j]
            <= inst.u * x[j]
        )
    # (B) assignment
    for i in range(n):
        h.addConstr(sum(y[i * m + j] for j in range(m)) == float(hs[i]))

    h.run()
    sol = h.getSolution()
    col = np.array(sol.col_value, dtype=float)
    # x and y are binary in K_s; snap the integral solution to remove the
    # solver's floating-point dirt (e.g. 4.8e-17) so downstream coefficients
    # stay clean for the LP/MIP builders.
    xval = np.round(col[[x[j].index for j in range(m)]])
    y0val = col[[y0[j].index for j in range(m)]]
    yval = np.round(
        col[[y[i * m + j].index for i in range(n) for j in range(m)]]
    ).reshape(n, m)
    theta = float(inst.q0 @ y0val - (inst.q * yval).sum())
    value = float(pi @ xval + pi0 * theta)
    return QStarResult(value=value, x=xval, theta=theta)
