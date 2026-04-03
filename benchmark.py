"""
Benchmark: Full Jacobian Descent Training Loop — JAX vs PyTorch (TorchJD)

Replicates the TorchJD "Basic Usage" example in both frameworks:
  - Model: Linear(10,5) -> ReLU -> Linear(5,2)
  - Two MSE losses on the two output columns
  - UPGrad aggregator for Jacobian descent
  - SGD optimizer (lr=0.1)

Compares: loss trajectories (accuracy) and wall-clock time (performance).
"""

import time
import sys
import os

import numpy as np

# --- JAX setup (must be before any jax import) ---
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# --- PyTorch setup ---
import torch
from torch.nn import Linear, MSELoss, ReLU, Sequential
from torch.optim import SGD

from torchjd import autojac
from torchjd.aggregation import UPGrad
from torchjd.autojac import jac_to_grad

# --- JAX UPGrad (from JaxJD.upgrad) ---
from JaxJD.upgrad import upgrad


# =============================================================================
# JAX Model + Training
# =============================================================================

def jax_init_params(w1_np, b1_np, w2_np, b2_np):
    """Initialize JAX params from numpy arrays (for weight-matching with PyTorch)."""
    return {
        "w1": jnp.array(w1_np),
        "b1": jnp.array(b1_np),
        "w2": jnp.array(w2_np),
        "b2": jnp.array(b2_np),
    }


def jax_forward(params, x):
    """Forward: Linear(10,5) -> ReLU -> Linear(5,2)."""
    h = x @ params["w1"].T + params["b1"]
    h = jnp.maximum(h, 0.0)
    out = h @ params["w2"].T + params["b2"]
    return out


def jax_loss1(params, x, target1):
    """MSE loss on first output column."""
    out = jax_forward(params, x)
    return jnp.mean((out[:, 0] - target1) ** 2)


def jax_loss2(params, x, target2):
    """MSE loss on second output column."""
    out = jax_forward(params, x)
    return jnp.mean((out[:, 1] - target2) ** 2)


PARAM_KEYS = ("w1", "b1", "w2", "b2")


def _flatten_grads(grad_tree):
    """Flatten a param dict of gradients into a single 1D vector."""
    return jnp.concatenate([grad_tree[k].ravel() for k in PARAM_KEYS])


def _unflatten_grads(flat, shapes_and_sizes):
    """Unflatten a 1D vector back into a param dict."""
    result = {}
    offset = 0
    for key, shape, size in shapes_and_sizes:
        result[key] = flat[offset:offset + size].reshape(shape)
        offset += size
    return result


@jax.jit
def jax_train_step(params, x, target1, target2, lr):
    """One step of Jacobian descent with UPGrad in JAX.

    1. Compute per-loss gradients (= rows of the Jacobian)
    2. Flatten & stack into Jacobian matrix (2 x total_params)
    3. Aggregate with UPGrad
    4. Unflatten, apply SGD update
    """
    grad1 = jax.grad(jax_loss1)(params, x, target1)
    grad2 = jax.grad(jax_loss2)(params, x, target2)

    flat1 = _flatten_grads(grad1)
    flat2 = _flatten_grads(grad2)
    jacobian = jnp.stack([flat1, flat2], axis=0)

    aggregated = upgrad(jacobian)

    shapes = [
        (k, params[k].shape, params[k].size) for k in PARAM_KEYS
    ]
    grad_dict = _unflatten_grads(aggregated, shapes)

    new_params = {k: params[k] - lr * grad_dict[k] for k in params}

    l1 = jax_loss1(new_params, x, target1)
    l2 = jax_loss2(new_params, x, target2)
    return new_params, l1, l2


# =============================================================================
# PyTorch Training
# =============================================================================

def pytorch_train_step(model, optimizer, aggregator, loss_fn, input_t, target1_t, target2_t):
    """One step of Jacobian descent with UPGrad in PyTorch (TorchJD)."""
    optimizer.zero_grad()
    output = model(input_t)
    loss1 = loss_fn(output[:, 0], target1_t)
    loss2 = loss_fn(output[:, 1], target2_t)
    autojac.backward([loss1, loss2])
    jac_to_grad(model.parameters(), aggregator)
    optimizer.step()
    with torch.no_grad():
        output = model(input_t)
        l1 = loss_fn(output[:, 0], target1_t).item()
        l2 = loss_fn(output[:, 1], target2_t).item()
    return l1, l2


# =============================================================================
# Main Comparison
# =============================================================================

def run_comparison(num_steps=100, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Generate shared data ---
    input_np = np.random.randn(16, 10).astype(np.float64)
    target1_np = np.random.randn(16).astype(np.float64)
    target2_np = np.random.randn(16).astype(np.float64)

    # --- PyTorch model ---
    torch.manual_seed(seed)
    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2)).double()
    optimizer = SGD(model.parameters(), lr=0.1)
    aggregator = UPGrad()
    loss_fn = MSELoss()

    input_t = torch.tensor(input_np)
    target1_t = torch.tensor(target1_np)
    target2_t = torch.tensor(target2_np)

    # Extract initial weights for JAX (match exactly)
    w1_np = model[0].weight.detach().numpy().copy()
    b1_np = model[0].bias.detach().numpy().copy()
    w2_np = model[2].weight.detach().numpy().copy()
    b2_np = model[2].bias.detach().numpy().copy()

    # --- JAX params (same initial weights) ---
    params = jax_init_params(w1_np, b1_np, w2_np, b2_np)
    input_jx = jnp.array(input_np)
    target1_jx = jnp.array(target1_np)
    target2_jx = jnp.array(target2_np)

    # --- Warmup JAX JIT ---
    params_warmup = {k: v.copy() for k, v in params.items()}
    _ = jax_train_step(params_warmup, input_jx, target1_jx, target2_jx, 0.1)

    # --- Training loops ---
    pt_losses1, pt_losses2 = [], []
    jx_losses1, jx_losses2 = [], []

    # PyTorch training
    t0 = time.perf_counter()
    for step in range(num_steps):
        l1, l2 = pytorch_train_step(model, optimizer, aggregator, loss_fn,
                                     input_t, target1_t, target2_t)
        pt_losses1.append(l1)
        pt_losses2.append(l2)
    pt_time = time.perf_counter() - t0

    # JAX training
    t0 = time.perf_counter()
    for step in range(num_steps):
        params, l1, l2 = jax_train_step(params, input_jx, target1_jx, target2_jx, 0.1)
        jx_losses1.append(float(l1))
        jx_losses2.append(float(l2))
    jx_time = time.perf_counter() - t0

    # --- Print results ---
    print("=" * 74)
    print("JACOBIAN DESCENT TRAINING: JAX (JaxJD) vs PyTorch (TorchJD)")
    print(f"Model: Linear(10,5)->ReLU->Linear(5,2) | Losses: 2x MSE | Steps: {num_steps}")
    print("=" * 74)

    print(f"\n{'Step':<8} {'PT Loss1':>10} {'PT Loss2':>10} {'JX Loss1':>10} {'JX Loss2':>10}"
          f" {'L1 Diff':>10} {'L2 Diff':>10}")
    print("-" * 68)
    milestones = [0, 1, 4, 9, 19, 49, 99] if num_steps >= 100 else list(range(min(num_steps, 10)))
    for i in milestones:
        if i < num_steps:
            d1 = abs(pt_losses1[i] - jx_losses1[i])
            d2 = abs(pt_losses2[i] - jx_losses2[i])
            print(f"{i+1:<8} {pt_losses1[i]:>10.6f} {pt_losses2[i]:>10.6f}"
                  f" {jx_losses1[i]:>10.6f} {jx_losses2[i]:>10.6f}"
                  f" {d1:>10.2e} {d2:>10.2e}")

    print(f"\nFinal losses after {num_steps} steps:")
    print(f"  PyTorch: L1={pt_losses1[-1]:.6f}, L2={pt_losses2[-1]:.6f}")
    print(f"  JAX:     L1={jx_losses1[-1]:.6f}, L2={jx_losses2[-1]:.6f}")
    final_d1 = abs(pt_losses1[-1] - jx_losses1[-1])
    final_d2 = abs(pt_losses2[-1] - jx_losses2[-1])
    print(f"  Diff:    L1={final_d1:.2e}, L2={final_d2:.2e}")

    # Timing
    print(f"\nTIMING:")
    print(f"  PyTorch: {pt_time*1000:.1f} ms total, {pt_time/num_steps*1000:.2f} ms/step")
    print(f"  JAX:     {jx_time*1000:.1f} ms total, {jx_time/num_steps*1000:.2f} ms/step")
    speedup = pt_time / jx_time if jx_time > 0 else float("inf")
    print(f"  JAX speedup: {speedup:.2f}x")

    # Parameter agreement
    w1_pt = model[0].weight.detach().numpy()
    b1_pt = model[0].bias.detach().numpy()
    w2_pt = model[2].weight.detach().numpy()
    b2_pt = model[2].bias.detach().numpy()

    print(f"\nPARAMETER AGREEMENT (after {num_steps} steps):")
    for name, pt_arr, jx_key in [
        ("Linear1.weight", w1_pt, "w1"),
        ("Linear1.bias", b1_pt, "b1"),
        ("Linear2.weight", w2_pt, "w2"),
        ("Linear2.bias", b2_pt, "b2"),
    ]:
        max_diff = np.max(np.abs(pt_arr - np.asarray(params[jx_key])))
        print(f"  {name:<16} max |diff| = {max_diff:.2e}")

    print()


if __name__ == "__main__":
    run_comparison(num_steps=100)
