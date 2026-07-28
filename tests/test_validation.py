"""Validation tests for the Lagrangian-cut reproduction.

These check the theoretical relationships from the paper on small instances:
    * z_LP  <=  restricted-Lagrangian bound  <=  z_LC(exact)  <=  z_IP  (Thm 3)
    * exact separation closes essentially the whole Lagrangian-dual gap.

Run with:  PYTHONPATH=src python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from lagcuts.algo2 import Config, run_algorithm2
from lagcuts.reference import solve_extensive_form
from lagcuts.sslp import generate_sslp


@pytest.fixture(scope="module")
def small_instance():
    return generate_sslp(m=4, n=6, S=4, k=1)


def _bound(inst, method, delta):
    cfg = Config(method=method, K=8, alpha=1.0, delta=delta, time_limit=60)
    return run_algorithm2(inst, cfg)


def test_bound_ordering(small_instance):
    inst = small_instance
    zIP = solve_extensive_form(inst)

    lp = _bound(inst, "benders", 0.0).final_bound
    zLC = _bound(inst, "exact", 0.0).final_bound

    # z_LP <= z_LC <= z_IP
    assert lp <= zLC + 1e-4
    assert zLC <= zIP + 1e-4
    # exact separation attains the Lagrangian-dual bound; for these tiny
    # instances that closes (almost) the entire gap to z_IP.
    assert (zIP - zLC) / abs(zIP) < 1e-3

    for method in ("rstr1", "rstr2", "rstrmip"):
        b = _bound(inst, method, 0.0).final_bound
        assert lp - 1e-3 <= b <= zLC + 1e-3


def test_benders_only_is_lp_bound(small_instance):
    log = _bound(small_instance, "benders", 0.0)
    assert log.lp_bound is not None
    assert abs(log.final_bound - log.lp_bound) < 1e-6
