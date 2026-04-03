"""
JAX implementation of UPGrad aggregator and UPGradWeighting.

Equivalent to torchjd.aggregation.UPGrad and torchjd.aggregation.UPGradWeighting.

Algorithm (from "Jacobian Descent For Multi-Objective Optimization", arXiv:2406.16232):
  1. Compute Gramian: G = J @ J.T
  2. Normalize: G / trace(G)
  3. Regularize: G + eps * I
  4. For each preference weight, solve QP: min 0.5*v^T G v  s.t. v >= u
  5. Sum projected weights, aggregate: result = w @ J

The QP is solved via Nesterov-accelerated projected gradient descent (pure JAX,
fully jit-compilable, no external solver dependencies).
"""

import functools

import jax
import jax.numpy as jnp


# =============================================================================
# QP Solver (internal)
# =============================================================================

def _solve_qp_nesterov(G, Gu, lr, u, num_iters):
    """Solve min 0.5*v^T G v  s.t. v >= u  via Nesterov accelerated PGD.

    Substituting s = v - u (s >= 0):
        min 0.5*s^T G s + (G u)^T s  s.t. s >= 0
    Converges as O(1/k^2) vs O(1/k) for vanilla PGD.

    :param G: PSD matrix (m, m).
    :param Gu: Precomputed G @ u, shape (m,).
    :param lr: Step size (1 / spectral_norm(G)).
    :param u: Constraint lower bound, shape (m,).
    :param num_iters: Number of iterations.
    :return: Projected weight vector v, shape (m,).
    """

    def body(_, state):
        s, s_prev, k = state
        t = (k - 1.0) / (k + 2.0)
        y = s + t * (s - s_prev)
        grad = G @ y + Gu
        s_new = jnp.maximum(0.0, y - lr * grad)
        return (s_new, s, k + 1.0)

    s0 = jnp.zeros_like(u)
    s_final, _, _ = jax.lax.fori_loop(0, num_iters, body, (s0, s0, 1.0))
    return u + s_final


def _project_weights(U, G, num_iters):
    """Project rows of U onto dual cone defined by G.

    For each row u of U, solves: min 0.5*v^T G v  s.t. v >= u

    :param U: Weight matrix (m, m), each row is a weight vector to project.
    :param G: Gramian matrix (m, m), symmetric positive definite.
    :param num_iters: Number of Nesterov PGD iterations.
    :return: Projected weight matrix (m, m).
    """
    lr = 1.0 / jnp.linalg.norm(G, ord=2)
    GU = G @ U.T
    W = jax.vmap(
        lambda u, gu: _solve_qp_nesterov(G, gu, lr, u, num_iters)
    )(U, GU.T)
    return W


# =============================================================================
# UPGradWeighting
# =============================================================================

@functools.partial(jax.jit, static_argnames=("num_iters",))
def upgrad_weighting(gramian, pref_vector=None, norm_eps=1e-4, reg_eps=1e-4, num_iters=50):
    """Compute UPGrad weight vector from Gramian.

    Equivalent to ``torchjd.aggregation.UPGradWeighting``.

    :param gramian: PSD Gramian matrix G = J @ J.T, shape (m, m).
    :param pref_vector: Preference vector of shape (m,). Defaults to uniform [1/m, ..., 1/m].
    :param norm_eps: Epsilon for normalization (avoid division by zero).
    :param reg_eps: Epsilon for regularization (ensure positive definiteness).
    :param num_iters: Number of Nesterov PGD iterations for QP solve.
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
    W = _project_weights(U, G_reg, num_iters)
    return jnp.sum(W, axis=0)


# =============================================================================
# UPGrad Aggregator
# =============================================================================

@functools.partial(jax.jit, static_argnames=("num_iters",))
def upgrad(J, pref_vector=None, norm_eps=1e-4, reg_eps=1e-4, num_iters=50):
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
    :param num_iters: Number of Nesterov PGD iterations for QP solve.
    :return: Aggregated gradient vector of shape (n,).
    """
    gramian = J @ J.T
    w = upgrad_weighting(gramian, pref_vector, norm_eps, reg_eps, num_iters)
    return w @ J
