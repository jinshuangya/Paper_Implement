"""Reference values for validation: the extensive-form optimum z_IP (eq. 3)."""

from __future__ import annotations

import highspy
import numpy as np

from .sslp import SSLPInstance


def solve_extensive_form(inst: SSLPInstance) -> float:
    """Solve the full extensive-form MIP and return z_IP."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    INF = highspy.kHighsInf
    n, m, S = inst.n, inst.m, inst.S

    x = h.addVariables(m, lb=0.0, ub=1.0)
    for v in x:
        h.setInteger(v)

    obj = sum(float(inst.c[j]) * x[j] for j in range(m))
    for s in range(S):
        hs = inst.h[s]
        y = h.addVariables(n * m, lb=0.0, ub=1.0)
        for v in y:
            h.setInteger(v)
        y0 = h.addVariables(m, lb=0.0, ub=INF)
        obj += float(inst.p[s]) * (
            sum(float(inst.q0[j]) * y0[j] for j in range(m))
            - sum(float(inst.q[i, j]) * y[i * m + j] for i in range(n) for j in range(m))
        )
        for j in range(m):
            h.addConstr(
                sum(float(inst.d[i, j]) * y[i * m + j] for i in range(n)) - y0[j]
                <= inst.u * x[j]
            )
        for i in range(n):
            h.addConstr(sum(y[i * m + j] for j in range(m)) == float(hs[i]))

    h.minimize(obj)
    h.run()
    return float(h.getObjectiveValue())
