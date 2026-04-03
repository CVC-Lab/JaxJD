"""
JAX implementation of UPGrad aggregator and UPGradWeighting.

Equivalent to torchjd.aggregation.UPGrad and torchjd.aggregation.UPGradWeighting.

Algorithm (from "Jacobian Descent For Multi-Objective Optimization", arXiv:2406.16232):
  1. Compute Gramian: G = J @ J.T
  2. Normalize: G / trace(G)
  3. Regularize: G + eps * I
  4. For each preference weight, solve QP: min 0.5*v^T G v  s.t. v >= u
  5. Sum projected weights, aggregate: result = w @ J

The QP is solved using qpax (primal-dual interior point method), which is a direct
solver equivalent to quadprog used by TorchJD. It is JIT-compilable, vmap-compatible,
and produces solutions exact to machine precision (~1e-12 in float64).
"""

import functools

import jax
import jax.numpy as jnp
import qpax


# =============================================================================
# QP Solver (internal)
# =============================================================================

def _solve_single_qp(G, u, solver_tol):
    """Solve min 0.5*v^T G v  s.t. v >= u  using qpax (interior point).

    This is the same QP that TorchJD solves via quadprog:
        solve_qp(G, zeros(m), -I, -u)

    :param G: PSD matrix (m, m).
    :param u: Constraint lower bound, shape (m,).
    :param solver_tol: Solver tolerance for qpax.
    :return: Solution v, shape (m,).
    """
    m = G.shape[0]
    q = jnp.zeros(m)
    G_ineq = -jnp.eye(m)   # -v <= -u  <==>  v >= u
    h_ineq = -u
    A_eq = jnp.zeros((0, m))
    b_eq = jnp.zeros(0)
    return qpax.solve_qp_primal(G, q, A_eq, b_eq, G_ineq, h_ineq, solver_tol=solver_tol)


def _project_weights(U, G, solver_tol):
    """Project rows of U onto dual cone defined by G.

    For each row u of U, solves: min 0.5*v^T G v  s.t. v >= u

    :param U: Weight matrix (m, m), each row is a weight vector to project.
    :param G: Gramian matrix (m, m), symmetric positive definite.
    :param solver_tol: Solver tolerance for qpax.
    :return: Projected weight matrix (m, m).
    """
    W = jax.vmap(lambda u: _solve_single_qp(G, u, solver_tol))(U)
    return W


# =============================================================================
# UPGradWeighting
# =============================================================================

@jax.jit
def upgrad_weighting(gramian, pref_vector=None, norm_eps=1e-4, reg_eps=1e-4, solver_tol=1e-10):
    """Compute UPGrad weight vector from Gramian.

    Equivalent to ``torchjd.aggregation.UPGradWeighting``.

    :param gramian: PSD Gramian matrix G = J @ J.T, shape (m, m).
    :param pref_vector: Preference vector of shape (m,). Defaults to uniform [1/m, ..., 1/m].
    :param norm_eps: Epsilon for normalization (avoid division by zero).
    :param reg_eps: Epsilon for regularization (ensure positive definiteness).
    :param solver_tol: Tolerance for the qpax QP solver.
    :return: Weight vector of shape (m,).
    """
    m = gramian.shape[0]

    pref = jnp.ones(m) / m if pref_vector is None else pref_vector
    U = jnp.diag(pref)

    # Normalize
    trace_G = jnp.trace(gramian)
    G_norm = jnp.where(trace_G < norm_eps, jnp.zeros_like(gramian), gramian / trace_G)

    # Regularize
    G_reg = G_norm + reg_eps * jnp.eye(m)

    # Project and sum
    W = _project_weights(U, G_reg, solver_tol)
    return jnp.sum(W, axis=0)


# =============================================================================
# UPGrad Aggregator
# =============================================================================

@jax.jit
def upgrad(J, pref_vector=None, norm_eps=1e-4, reg_eps=1e-4, solver_tol=1e-10):
    """Aggregate a Jacobian matrix using UPGrad.

    Equivalent to ``torchjd.aggregation.UPGrad``.

    Projects each row of J onto the dual cone of all rows, combines the results
    with a preference vector (default: uniform average). Guarantees no loss is
    negatively affected by the update.

    :param J: Jacobian matrix of shape (m, n), where m = number of objectives,
        n = number of parameters.
    :param pref_vector: Preference vector of shape (m,). Defaults to uniform.
    :param norm_eps: Epsilon for normalization.
    :param reg_eps: Epsilon for regularization.
    :param solver_tol: Tolerance for the qpax QP solver.
    :return: Aggregated gradient vector of shape (n,).
    """
    gramian = J @ J.T
    w = upgrad_weighting(gramian, pref_vector, norm_eps, reg_eps, solver_tol)
    return w @ J
