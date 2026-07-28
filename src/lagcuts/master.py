"""Benders master problem (root-node LP relaxation) with dynamic cuts.

The master is

    min  c @ x + sum_s p_s theta_s
    s.t. pi @ x + pi0 * theta_s >= rhs      for every cut of scenario s
         0 <= x <= 1                        (LP relaxation for the root-node study)

Both Benders cuts (pi0 = 1) and Lagrangian cuts (general pi0 >= 0) share this
shape.  The model is kept persistent and rows are appended incrementally so the
objective value after each solve is a valid lower bound on z_IP.
"""

from __future__ import annotations

import highspy
import numpy as np

from .sslp import SSLPInstance


class BendersMaster:
    def __init__(self, inst: SSLPInstance, integer: bool = False):
        self.inst = inst
        m, S = inst.m, inst.S
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        INF = highspy.kHighsInf

        self.x = h.addVariables(m, lb=0.0, ub=1.0)
        if integer:
            for v in self.x:
                h.setInteger(v)
        # theta_s free below (bounded once cuts are added)
        self.theta = h.addVariables(S, lb=-INF, ub=INF)

        h.minimize(
            sum(float(inst.c[j]) * self.x[j] for j in range(m))
            + sum(float(inst.p[s]) * self.theta[s] for s in range(S))
        )
        self.h = h
        self.n_cuts = np.zeros(S, dtype=int)

    def add_cut(self, s: int, pi: np.ndarray, pi0: float, rhs: float) -> bool:
        """Append  pi @ x + pi0 * theta_s >= rhs  for scenario s.

        Returns False (skipping) if the cut is numerically degenerate: any
        non-finite entry, or a left-hand side with no meaningful variable
        coefficient (HiGHS rejects an empty row)."""
        m = self.inst.m
        pi = np.asarray(pi, dtype=float)
        if not (np.all(np.isfinite(pi)) and np.isfinite(pi0) and np.isfinite(rhs)):
            return False
        coeff_tol = 1e-9
        expr = None
        for j in range(m):
            if abs(pi[j]) > coeff_tol:
                term = float(pi[j]) * self.x[j]
                expr = term if expr is None else expr + term
        if abs(pi0) > coeff_tol:
            term = float(pi0) * self.theta[s]
            expr = term if expr is None else expr + term
        if expr is None:  # nothing on the LHS -> not a usable cut
            return False
        self.h.addConstr(expr >= float(rhs))
        self.n_cuts[s] += 1
        return True

    def solve(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (x, theta, objective).  Objective is a valid lower bound."""
        self.h.run()
        col = np.array(self.h.getSolution().col_value, dtype=float)
        m, S = self.inst.m, self.inst.S
        xval = col[[self.x[j].index for j in range(m)]]
        thval = col[[self.theta[s].index for s in range(S)]]
        return xval, thval, float(self.h.getObjectiveValue())
