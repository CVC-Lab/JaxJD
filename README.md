# JaxJD: UPGrad for JAX

A JAX implementation of the **UPGrad** (Unconflicting Projection of Gradients) aggregator
from the paper [*Jacobian Descent for Multi-Objective Optimization*](https://arxiv.org/abs/2406.16232)
(Quinton & Rey, 2024).

This is a JAX equivalent of [`torchjd.aggregation.UPGrad`](https://github.com/TorchJD/torchjd).
Two QP solvers are available:
- **qpax** (default): Primal-dual interior point method. Direct solver equivalent to
  `quadprog` used by TorchJD --- exact to machine precision (~1e-12 in float64).
- **nesterov_pgd**: Nesterov-accelerated projected gradient descent. Pure JAX with no
  external dependencies --- accurate to ~1e-10, faster on small problems.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Full Training Example Step by Step](#3-full-training-example-step-by-step)
4. [Algorithm](#4-algorithm)
5. [Performance: JaxJD vs TorchJD](#5-performance-jaxjd-vs-torchjd)

---

## 1. Installation

JaxJD requires JAX. The default solver also requires qpax:

```bash
pip install jax jaxlib qpax
```

If you only want the `nesterov_pgd` solver (no extra dependencies beyond JAX):

```bash
pip install jax jaxlib
```

Then import from the `JaxJD` folder:

```python
from JaxJD.upgrad import upgrad, upgrad_weighting
```

---

## 2. Quick Start

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from JaxJD.upgrad import upgrad

# A Jacobian matrix: 2 objectives, 3 parameters
J = jnp.array([[-4.0, 1.0, 1.0],
               [ 6.0, 1.0, 1.0]])

# Aggregate using qpax (default, exact)
result = upgrad(J)
print(result)  # [0.2929, 1.9004, 1.9004]

# Or use Nesterov PGD (no extra dependencies, slightly less precise)
result = upgrad(J, solver="nesterov_pgd")
print(result)  # [0.2929, 1.9004, 1.9004]
```

That is it. One function call. The result is a single gradient vector that respects both
objectives --- neither loss will get worse if you step in this direction.

---

## 3. Full Training Example Step by Step

This section walks through a complete training loop. We will train a small neural network
that has **two objectives** (two loss functions) and use UPGrad to combine their gradients
at every step.

### 3.1. The Problem

We have:
- A neural network: `Linear(10, 5) -> ReLU -> Linear(5, 2)` (two output columns)
- Input: a batch of 16 vectors, each of length 10
- Two targets: one for each output column
- Two MSE losses: one per target

The challenge is: how do we update the model so that **both** losses go down? If the
gradients from the two losses point in different directions (they conflict), a naive
average might make one loss worse. UPGrad solves this.

### 3.2. Setup

```python
import jax
jax.config.update("jax_enable_x64", True)  # Use float64 for precision
import jax.numpy as jnp
import numpy as np
from JaxJD.upgrad import upgrad
```

### 3.3. Define the Model

In JAX, there are no `nn.Module` classes. Instead, parameters are plain dictionaries
and the forward pass is a pure function.

```python
def init_params(key):
    """Create random initial weights."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "w1": jax.random.normal(k1, (5, 10)) * 0.1,   # Linear layer 1: 10 -> 5
        "b1": jnp.zeros(5),
        "w2": jax.random.normal(k3, (2, 5)) * 0.1,    # Linear layer 2: 5 -> 2
        "b2": jnp.zeros(2),
    }

def forward(params, x):
    """Forward pass: Linear -> ReLU -> Linear."""
    h = x @ params["w1"].T + params["b1"]   # Shape: (batch, 5)
    h = jnp.maximum(h, 0.0)                 # ReLU
    out = h @ params["w2"].T + params["b2"] # Shape: (batch, 2)
    return out
```

### 3.4. Define the Losses

Each loss function takes the full parameter dictionary and returns a scalar.

```python
def loss1(params, x, target1):
    """MSE loss on the first output column."""
    out = forward(params, x)
    return jnp.mean((out[:, 0] - target1) ** 2)

def loss2(params, x, target2):
    """MSE loss on the second output column."""
    out = forward(params, x)
    return jnp.mean((out[:, 1] - target2) ** 2)
```

### 3.5. Build the Jacobian

This is the key step. In regular gradient descent, you compute one gradient. In Jacobian
descent, you compute **one gradient per loss** and stack them into a matrix.

```python
# jax.grad computes the gradient of a scalar function w.r.t. its first argument
grad1 = jax.grad(loss1)(params, x, target1)  # gradient from loss 1
grad2 = jax.grad(loss2)(params, x, target2)  # gradient from loss 2
```

Each `grad` is a dictionary with the same structure as `params` (one array per layer).
We flatten them into 1D vectors and stack them as rows:

```python
PARAM_KEYS = ("w1", "b1", "w2", "b2")

def flatten(grad_dict):
    return jnp.concatenate([grad_dict[k].ravel() for k in PARAM_KEYS])

flat1 = flatten(grad1)  # Shape: (total_params,)
flat2 = flatten(grad2)  # Shape: (total_params,)

jacobian = jnp.stack([flat1, flat2], axis=0)  # Shape: (2, total_params)
```

Now `jacobian` is a 2-by-n matrix. Row 0 is the gradient of loss 1. Row 1 is the
gradient of loss 2.

### 3.6. Aggregate with UPGrad

```python
aggregated = upgrad(jacobian)  # default: solver="qpax" (exact)

# Or with Nesterov PGD:
# aggregated = upgrad(jacobian, solver="nesterov_pgd")
```

This single line does all the math (see [Algorithm](ALGORITHM.md) for details). The
output is a single gradient vector that is guaranteed to not conflict with either loss.

### 3.7. Update the Parameters (SGD)

Unflatten the aggregated gradient back into the parameter shape and subtract:

```python
def unflatten(flat, params):
    result = {}
    offset = 0
    for k in PARAM_KEYS:
        size = params[k].size
        result[k] = flat[offset:offset + size].reshape(params[k].shape)
        offset += size
    return result

lr = 0.1
grad_dict = unflatten(aggregated, params)
params = {k: params[k] - lr * grad_dict[k] for k in params}
```

### 3.8. Putting It All Together (JIT-Compiled)

For best performance, wrap the entire step in `@jax.jit`. This compiles the whole thing
--- forward pass, both gradients, UPGrad aggregation, and SGD update --- into a single
optimized kernel.

```python
@jax.jit
def train_step(params, x, target1, target2, lr):
    # Step 1: Compute per-loss gradients
    grad1 = jax.grad(loss1)(params, x, target1)
    grad2 = jax.grad(loss2)(params, x, target2)

    # Step 2: Build Jacobian matrix (2 x total_params)
    jacobian = jnp.stack([flatten(grad1), flatten(grad2)], axis=0)

    # Step 3: Aggregate with UPGrad
    aggregated = upgrad(jacobian)

    # Step 4: SGD update
    grad_dict = unflatten(aggregated, params)
    new_params = {k: params[k] - lr * grad_dict[k] for k in params}
    return new_params
```

### 3.9. Training Loop

```python
key = jax.random.PRNGKey(42)
params = init_params(key)

# Generate some data
x = jax.random.normal(key, (16, 10))
target1 = jax.random.normal(key, (16,))
target2 = jax.random.normal(key, (16,))

# Train for 100 steps
for step in range(100):
    params = train_step(params, x, target1, target2, lr=0.1)

    if step % 20 == 0:
        l1 = loss1(params, x, target1)
        l2 = loss2(params, x, target2)
        print(f"Step {step:3d}  Loss1={l1:.4f}  Loss2={l2:.4f}")
```

Both losses should decrease together, without either one increasing.

---

## 4. Algorithm

For a detailed explanation of the math behind UPGrad --- including the dual cone,
the Gramian trick, the QP formulation, convergence guarantees, and how each line of
code maps to the equations --- see:

**[ALGORITHM.md](ALGORITHM.md)**

**Short summary:** UPGrad projects each gradient onto the dual cone of all gradients
(the set of "safe" directions that do not worsen any loss), then averages the results.
The projection is computed efficiently by solving a small quadratic program (QP) in
m-dimensional space (m = number of losses) using the Gramian matrix G = J J^T, rather
than working in n-dimensional parameter space.

JaxJD provides two QP solvers:

| Solver | Method | Precision | Dependencies | Best for |
|---|---|---|---|---|
| `"qpax"` (default) | Primal-dual interior point | Exact (~1e-12) | `qpax` | Matching TorchJD exactly |
| `"nesterov_pgd"` | Nesterov accelerated PGD | ~1e-10 | None (pure JAX) | Zero-dependency setups |

Both are JIT-compilable and vmap-compatible.

---

## 5. Performance: JaxJD vs TorchJD

We benchmarked both implementations on the same hardware (CPU) with the same inputs.
Both use direct QP solvers: TorchJD uses `quadprog` (active set), JaxJD uses `qpax`
(interior point). Both produce exact solutions.

### 5.1. Accuracy: Full Training Loop

**Experiment:** Both frameworks train the same model (Linear(10,5)->ReLU->Linear(5,2))
from identical initial weights and data: batch of 16 inputs, two MSE losses, SGD with
lr=0.1. We run 100 training steps and compare losses and parameters at each milestone.

**Loss trajectory comparison:**

| Step | TorchJD Loss1 | TorchJD Loss2 | JaxJD Loss1 | JaxJD Loss2 | L1 Diff | L2 Diff |
|---|---|---|---|---|---|---|
| 1 | 0.580639 | 1.062991 | 0.580639 | 1.062991 | 1.95e-11 | 7.38e-12 |
| 2 | 0.546955 | 1.043270 | 0.546955 | 1.043270 | 1.94e-11 | 3.95e-12 |
| 5 | 0.465893 | 1.008172 | 0.465893 | 1.008172 | 1.75e-11 | 2.64e-11 |
| 10 | 0.379066 | 0.960707 | 0.379066 | 0.960707 | 1.15e-11 | 2.53e-11 |
| 20 | 0.295198 | 0.818730 | 0.295198 | 0.818730 | 3.99e-12 | 4.64e-11 |
| 50 | 0.172077 | 0.356759 | 0.172077 | 0.356759 | 2.24e-12 | 9.39e-12 |
| 100 | 0.102555 | 0.226943 | 0.102555 | 0.226943 | 1.28e-12 | 5.01e-12 |

Loss differences are ~1e-12 (trillionths) at every step. Both use direct QP solvers, so
the only difference is floating-point rounding from different operation ordering between
JAX and PyTorch.

**Final parameter agreement (after 100 steps):**

| Parameter | Shape | Max |diff| |
|---|---|---|
| Linear1.weight | (5, 10) | 1.80e-11 |
| Linear1.bias | (5,) | 1.05e-11 |
| Linear2.weight | (2, 5) | 2.01e-11 |
| Linear2.bias | (2,) | 1.61e-11 |

The largest parameter difference is 2.01e-11 after 100 steps. The two implementations
are numerically identical for all practical purposes.

### 5.2. Speed: Full Training Step

**Experiment:** Same setup as 5.1. Each training step includes: forward pass, computing
two per-loss gradients, building the Jacobian, aggregating with UPGrad, and applying the
SGD update. We run 100 steps and measure total wall-clock time.

| | Total (100 steps) | Per step | Speedup |
|---|---|---|---|
| TorchJD | 1398 ms | 13.98 ms | --- |
| JaxJD | 791 ms | 7.91 ms | **1.77x** |

JaxJD is 1.77x faster because `@jax.jit` compiles the entire training step --- forward
pass, gradient computation, QP solve, and SGD update --- into a single optimized XLA
kernel.

### 5.3. Why JaxJD is Faster

1. **JIT compilation**: The entire pipeline is compiled into a single optimized XLA
   kernel. There is no Python-to-C++ round-trip per operation.

2. **vmap parallelism**: All m QP solves run in parallel via `jax.vmap`, whereas TorchJD
   solves them one at a time in a Python loop through numpy.

3. **No framework crossing**: TorchJD converts tensors to numpy, calls `quadprog`, and
   converts back. JaxJD stays in JAX arrays the entire time.

4. **Whole-step JIT**: When you wrap the training step in `@jax.jit`, JAX compiles
   everything --- forward pass, gradient computation, UPGrad, and SGD update --- into
   one fused kernel. TorchJD cannot do this because the QP solver is outside PyTorch's
   computation graph.

### 5.4. When to Use Which

| Scenario | Recommendation |
|---|---|
| Already using JAX (Flax, Equinox, Optax) | JaxJD --- stays in the JAX ecosystem |
| Already using PyTorch (Lightning, etc.) | TorchJD --- no framework switch needed |
| Need GPU acceleration for UPGrad itself | JaxJD --- the QP solver runs on GPU too |
| Instance-wise risk minimization (m = batch size) | JaxJD --- m can be 32-128, big speedup |

---

## References

- Quinton, P. & Rey, V. (2024). *Jacobian Descent for Multi-Objective Optimization*.
  arXiv:2406.16232. [Paper](https://arxiv.org/abs/2406.16232)
- Tracy, K. & Manchester, Z. (2024). *On the Differentiability of the Primal-Dual
  Interior-Point Method*. arXiv:2406.11749. [qpax](https://github.com/kevin-tracy/qpax)
- TorchJD: [github.com/TorchJD/torchjd](https://github.com/TorchJD/torchjd)
