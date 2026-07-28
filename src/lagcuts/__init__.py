"""Lagrangian-cut generation for two-stage stochastic integer programs.

Reproduction of Chen & Luedtke (2022), "On Generating Lagrangian Cuts for
Two-Stage Stochastic Integer Programs" (arXiv:2106.04023v2), plus a research
scaffold for extending the restricted-separation idea along two general axes:
adaptive low-dimensional coefficient subspaces and amortised (learned)
separation across scenarios.
"""

from .sslp import SSLPInstance, generate_sslp

__all__ = ["SSLPInstance", "generate_sslp"]
