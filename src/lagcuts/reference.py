"""Reference values for validation: the extensive-form optimum z_IP (eq. 3)."""

from __future__ import annotations

import highspy
import numpy as np

from .oracle import _add_second_stage, _second_stage_cost
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
        y = h.addVariables(n * m, lb=0.0, ub=1.0)
        for v in y:
            h.setInteger(v)
        y0 = h.addVariables(m, lb=0.0, ub=INF)
        obj += float(inst.p[s]) * _second_stage_cost(inst, y, y0)
        _add_second_stage(h, inst, s, y, y0, x, x_is_var=True)

    h.minimize(obj)
    h.run()
    return float(h.getObjectiveValue())
