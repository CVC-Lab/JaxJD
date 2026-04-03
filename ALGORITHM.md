# The Math Behind UPGrad

This document explains the UPGrad algorithm. It covers
the problem UPGrad solves, the mathematics behind it, and how each piece of code in
`upgrad.py` maps to the math.

Based on the paper: [*Jacobian Descent for Multi-Objective Optimization*](https://arxiv.org/abs/2406.16232)
(Quinton & Rey, 2024).

---

## Table of Contents

1. [The Problem: Conflicting Gradients](#1-the-problem-conflicting-gradients)
2. [The Jacobian Matrix](#2-the-jacobian-matrix)
3. [The Dual Cone](#3-the-dual-cone)
4. [What UPGrad Does](#4-what-upgrad-does)
5. [The Efficient Formulation (Dual QP)](#5-the-efficient-formulation-dual-qp)
6. [Step-by-Step Algorithm](#6-step-by-step-algorithm)
7. [How the QP is Solved](#7-how-the-qp-is-solved)
8. [Three Guarantees](#8-three-guarantees)
9. [Convergence to the Pareto Front](#9-convergence-to-the-pareto-front)
10. [How the Code Maps to the Math](#10-how-the-code-maps-to-the-math)

---

## 1. The Problem: Conflicting Gradients

Suppose you are training a model with two loss functions, L1 and L2. Each loss produces
its own gradient:

- g1 = gradient of L1 (tells you how to move to reduce L1)
- g2 = gradient of L2 (tells you how to move to reduce L2)

The simplest thing to do is average them: `g = (g1 + g2) / 2`. But what if g1 and g2
point in opposite directions? Then the average might actually **increase** one of the
losses. This is called a **conflict**.

**Example:** Imagine you are standing on a hill. Loss 1 says "go north" and loss 2 says
"go south". The average says "stay still" --- but that is not helpful. What you really
want is to move in a direction that helps both, or at least does not hurt either one.
Maybe "go east" helps both a little. UPGrad finds that direction.

---

## 2. The Jacobian Matrix

Instead of working with individual gradients, we stack them into a matrix called the
**Jacobian**:

```
J = [ g1 ]    (row 0 = gradient of loss 1)
    [ g2 ]    (row 1 = gradient of loss 2)
```

If we have m losses and n parameters, J is an m-by-n matrix. Each row is one gradient
vector.

In regular gradient descent (one loss), you have a single gradient vector. In Jacobian
descent (multiple losses), you have a matrix of gradients --- one row per loss.

---

## 3. The Dual Cone

The **dual cone** of the rows of J is the set of all directions that do not conflict
with any gradient. A direction d is in the dual cone if:

```
g1 . d >= 0   (d does not make loss 1 worse)
g2 . d >= 0   (d does not make loss 2 worse)
```

where `.` is the dot product. In other words, the dual cone is the set of all "safe"
directions --- directions where every loss either stays the same or improves.

**Geometric picture:** If you draw the two gradients g1 and g2 as arrows, the dual
cone is the wedge-shaped region where both arrows point "forward" (the dot product
with d is non-negative). Any direction inside this wedge will not make either loss worse.

If the two gradients point in roughly the same direction, the dual cone is wide --- many
safe directions exist. If they point in opposite directions, the dual cone is narrow ---
only a few directions are safe. If they are exactly opposite, the dual cone is a single
line perpendicular to both.

---

## 4. What UPGrad Does

UPGrad takes each gradient g_i and **projects** it onto the dual cone. The projection
finds the closest point in the dual cone to g_i:

```
projected_g1 = the point in the dual cone closest to g1
projected_g2 = the point in the dual cone closest to g2
```

"Closest" means the shortest Euclidean distance. The projection pushes g_i just enough
to make it safe (non-conflicting with all other gradients), but keeps it as close to the
original gradient as possible.

Then it averages the projections:

```
result = (projected_g1 + projected_g2) / m
```

Since both projected vectors are in the dual cone, and the dual cone is convex (any
average of points in the cone is also in the cone), the result is also in the dual cone.
This means: **the final update direction does not conflict with any loss**.

If the gradients already agree (no conflict), the projections do not change anything,
and UPGrad just returns the simple average. UPGrad only modifies the gradients when
there is an actual conflict.

---

## 5. The Efficient Formulation (Dual QP)

Computing the projection directly would require working in n-dimensional space (where n
is the number of parameters --- often millions). This is too expensive.

Instead, UPGrad uses a mathematical trick to work in m-dimensional space (where m is the
number of losses --- usually 2 to 32). The key insight (Proposition 1 from the paper) is:

> For any vector u in m-dimensional space, the projection of J^T u onto the dual cone
> of the rows of J equals J^T w, where w solves:
>
> **minimize** v^T G v **subject to** v >= u (element-wise)
>
> where G = J J^T is the **Gramian** matrix (m x m).

This is called a **quadratic program** (QP). It has only m variables (one per loss),
not n variables (one per parameter). Since m is tiny compared to n, this is very fast.

**Why this works:** The Gramian G = J J^T captures all the information about how the
gradients relate to each other. Entry G[i,j] = dot(g_i, g_j) tells you how aligned
gradients i and j are. You do not need the full n-dimensional gradients to compute the
projection --- you only need their pairwise dot products.

---

## 6. Step-by-Step Algorithm

Here is the full algorithm that `upgrad(J)` runs:

### Step 1: Compute the Gramian

```
G = J @ J.T
```

G is an m-by-m matrix. Entry G[i,j] = dot(g_i, g_j), which measures how aligned
gradients i and j are. If G[i,j] > 0, gradients i and j agree. If G[i,j] < 0, they
conflict. The diagonal entries G[i,i] = ||g_i||^2 are the squared norms of each gradient.

### Step 2: Normalize

```
G_norm = G / trace(G)
```

The trace of G equals the sum of the diagonal entries, which is the sum of the squared
norms of all gradients: trace(G) = ||g1||^2 + ||g2||^2 + ... + ||gm||^2.

Dividing by the trace makes the algorithm **scale-invariant** --- multiplying all
gradients by a constant does not change the result. This is important because the scale
of gradients can vary widely during training.

If trace(G) is nearly zero (all gradients are nearly zero), we return zeros --- there is
nothing to do.

### Step 3: Regularize

```
G_reg = G_norm + eps * I
```

Adding a tiny multiple of the identity matrix (eps = 0.0001 by default) ensures that
G_reg is strictly positive definite. Due to floating-point arithmetic, the computed
Gramian might have tiny negative eigenvalues (e.g., -1e-17 instead of 0). Adding eps * I
shifts all eigenvalues up by eps, making them all positive. This prevents the QP solver
from failing.

### Step 4: Set up the preference weights

```
U = I / m      (m-by-m identity matrix divided by m)
```

Each row of U is a standard basis vector scaled by 1/m.

- Row 0: [1/m, 0, 0, ..., 0] --- represents "project the 1st gradient"
- Row 1: [0, 1/m, 0, ..., 0] --- represents "project the 2nd gradient"
- Row i: the i-th standard basis vector scaled by 1/m

The 1/m factor means we will average the projections equally. You can also pass a
custom `pref_vector` to weight some losses more than others.

### Step 5: Solve one QP per gradient

For each row u of U (i.e., for each gradient), solve:

```
minimize    v^T G_reg v
subject to  v >= u     (element-wise)
```

This finds the weight vector v that produces the smallest-norm combination of gradients
(minimizes v^T G v = ||J^T v||^2) while satisfying the dual cone constraint v >= u.

The constraint v >= u (element-wise) means every component of v must be at least as
large as the corresponding component of u. Since u = e_i / m, this means v_i >= 1/m and
v_j >= 0 for all j != i. The solution v will increase some components beyond u to avoid
conflicts.

All m QP solves are run **in parallel** using `jax.vmap`.

### Step 6: Sum the projected weights

```
w = sum of all rows of W     (where W[i] = solution of QP for gradient i)
```

Each row of W is the weight vector from one QP solve. Summing them gives the combined
weight vector for the final aggregation.

### Step 7: Compute the final aggregation

```
result = w @ J
```

This is a weighted combination of the gradient rows: result = w1*g1 + w2*g2 + ... + wm*gm.
The weights w come from the QP solutions. The result is guaranteed to be in the dual cone.

---

## 7. How the QP is Solved

Both TorchJD and JaxJD solve the exact same quadratic program:

```
minimize    0.5 * v^T G v
subject to  -v <= -u     (i.e., v >= u, element-wise)
```

Three solvers are available across TorchJD and JaxJD:

### TorchJD: quadprog (Active Set Method)

TorchJD uses the `quadprog` library through numpy. It works by:

1. Starting with a guess of which constraints are "active" (v_i = u_i, meaning the
   constraint is tight) versus "inactive" (v_i > u_i, meaning the constraint is loose).
2. Solving the linear system defined by the KKT (Karush-Kuhn-Tucker) conditions for
   that active set.
3. Checking if any constraint is violated or any active constraint should be released.
4. Adding or removing one constraint from the active set.
5. Repeating until no changes are needed.

Each iteration solves an exact linear system. When it terminates, the answer is exact.
However, quadprog is a C library called through numpy --- it cannot be JIT-compiled by
JAX or run on GPU.

### JaxJD solver="qpax" (Primal-Dual Interior Point Method) --- default

JaxJD's default solver uses the `qpax` library, a JAX-native primal-dual interior point
solver. It works by:

1. Starting at a point strictly inside all constraints (v_i > u_i for all i).
2. Adding a logarithmic barrier: -mu * sum(log(v_i - u_i)) that penalizes getting
   close to any constraint boundary.
3. Solving for the minimum of (objective + barrier) using Newton's method.
4. Reducing mu (weakening the barrier).
5. Repeating --- as mu approaches 0, the solution approaches the true optimum.

This is a **direct solver** like quadprog. Both produce exact solutions that agree to
~1e-12 in float64. The remaining tiny difference is just from different floating-point
operation ordering, not from algorithmic approximation.

Usage: `upgrad(J)` or `upgrad(J, solver="qpax")`

### JaxJD solver="nesterov_pgd" (Nesterov Accelerated Projected Gradient Descent)

The alternative solver is pure JAX with no external dependencies. It works by:

1. Substituting s = v - u (so s >= 0), transforming the QP into:
   minimize 0.5 * s^T G s + (G u)^T s  subject to s >= 0
2. Computing the optimal step size: lr = 1 / spectral_norm(G)
3. Running Nesterov-accelerated projected gradient descent for 50 iterations:
   - Compute momentum term: y = s + t * (s - s_prev)
   - Gradient step: s_new = max(0, y - lr * (G y + G u))
4. Returning v = u + s

The `max(0, ...)` is the projection onto s >= 0 (ensuring v >= u). Nesterov momentum
gives O(1/k^2) convergence --- much faster than vanilla gradient descent's O(1/k).

This solver is **iterative**, so it does not give the exact answer. However, after 50
iterations on the well-conditioned regularized Gramian, the accuracy is ~1e-10 to 1e-14,
which is more than sufficient for training. It is faster than qpax for small problems
because it avoids the overhead of Newton step Cholesky factorizations.

Usage: `upgrad(J, solver="nesterov_pgd")`

### Solver comparison

| | quadprog (TorchJD) | qpax (JaxJD default) | nesterov_pgd (JaxJD) |
|---|---|---|---|
| Type | Direct (active set) | Direct (interior point) | Iterative (gradient) |
| Precision | ~1e-16 | ~1e-12 | ~1e-10 |
| JIT / vmap | No | Yes | Yes |
| GPU | No | Yes | Yes |
| Dependencies | quadprog, numpy | qpax | None (pure JAX) |
| Speed (small m) | Fast | Moderate | Fast |
| Speed (large m) | Slow | Moderate | Fast |

Both JaxJD solvers are JIT-compilable and vmap-compatible in JAX. This means:
- All m QP solves run in parallel (via `jax.vmap`)
- The entire UPGrad pipeline compiles into a single optimized XLA kernel (via `jax.jit`)
- They can run on GPU

---

## 8. Three Guarantees

UPGrad is the only aggregator that satisfies all three properties simultaneously:

### Non-conflicting

The output never makes any individual loss worse. Formally:

```
dot(g_i, result) >= 0    for every gradient g_i in the Jacobian
```

This means: if you step in the direction of the UPGrad output, every loss either
decreases or stays the same (for a sufficiently small step size).

**Why:** Each projected gradient is in the dual cone, the dual cone is convex, and
their sum is also in the dual cone. Every vector in the dual cone satisfies the
non-conflicting property by definition.

### Linear under scaling

If you multiply a gradient by a positive constant c (because one loss is larger), the
result scales proportionally. Formally:

```
A(diag(c) * J) is linear in c    for any c > 0
```

This means: big gradients get proportionally bigger influence, just like in standard
gradient descent. If loss 1 has a gradient 10x larger than loss 2, loss 1 will have
more influence on the update --- which is the natural behavior.

**Why:** The projection onto a closed convex cone satisfies pi(a * x) = a * pi(x) for
any positive scalar a. Since UPGrad projects each scaled gradient separately, the
linearity carries through.

### Weighted

The output can always be written as a weighted combination of the original gradients:

```
result = w1*g1 + w2*g2 + ... + wm*gm
```

for some weight vector w. This means the update stays in the span of the gradients ---
it does not invent new directions. It only adjusts how much each gradient contributes.

**Why:** By Proposition 1 of the paper, the projection of J^T u equals J^T w, which is
exactly a weighted combination of the rows of J.

---

## 9. Convergence to the Pareto Front

Theorem 1 from the paper proves that if the objective function f is:
- **Smooth** (Lipschitz continuous Jacobian)
- **Convex** (in the component-wise partial order)

Then Jacobian descent with UPGrad converges to the **Pareto front** --- the set of points
where no single objective can be improved without worsening another.

**What is the Pareto front?** A point x is Pareto optimal if there is no other point y
such that f_i(y) <= f_i(x) for all losses and f_j(y) < f_j(x) for at least one loss.
The Pareto front is the image f(X*) of the set of all Pareto optimal points.

UPGrad is the first non-conflicting aggregator with this convergence guarantee. Other
aggregators either converge only to weaker notions of stationarity (e.g., weak Pareto
stationarity) or do not have convergence guarantees at all.

---

## 10. How the Code Maps to the Math

Here is how each function in `upgrad.py` corresponds to the algorithm above:

| Algorithm Step | Code in `upgrad.py` | What It Does |
|---|---|---|
| G = J J^T | `gramian = J @ J.T` | Compute the Gramian matrix |
| G / trace(G) | `G / jnp.trace(G)` | Normalize for scale invariance |
| G + eps * I | `G_norm + reg_eps * jnp.eye(m)` | Regularize for numerical stability |
| U = I / m | `jnp.eye(m) / m` | Set up uniform preference weights |
| Solve QP (qpax) | `_solve_qp_qpax(G, u, solver_tol)` | Direct QP solve (interior point) |
| Solve QP (nesterov) | `_solve_qp_nesterov(G, Gu, lr, u, num_iters)` | Iterative QP solve (PGD) |
| Parallelize | `jax.vmap(...)` | Solve all m QPs simultaneously |
| w = sum(W) | `jnp.sum(W, axis=0)` | Combine projected weights |
| result = w @ J | `w @ J` | Final weighted combination |

### Public API

**`upgrad(J, solver="qpax")`** --- Takes a Jacobian matrix (m, n) and returns the
aggregated gradient vector (n,). This is the equivalent of `torchjd.aggregation.UPGrad`.

```python
from JaxJD.upgrad import upgrad

J = jnp.array([[-4.0, 1.0, 1.0],
               [ 6.0, 1.0, 1.0]])

# Default: exact solver (requires qpax package)
result = upgrad(J)                          # [0.2929, 1.9004, 1.9004]

# Alternative: pure JAX, no extra dependencies
result = upgrad(J, solver="nesterov_pgd")   # [0.2929, 1.9004, 1.9004]
```

**`upgrad_weighting(gramian, solver="qpax")`** --- Takes a Gramian matrix (m, m) and
returns the weight vector (m,). This is the equivalent of
`torchjd.aggregation.UPGradWeighting`. Use this if you already have the Gramian and
want just the weights.

```python
from JaxJD.upgrad import upgrad_weighting

G = J @ J.T
w = upgrad_weighting(G)                          # [1.1109, 0.7894]
w = upgrad_weighting(G, solver="nesterov_pgd")   # [1.1109, 0.7894]
```

---

## References

- Quinton, P. & Rey, V. (2024). *Jacobian Descent for Multi-Objective Optimization*.
  arXiv:2406.16232. [Paper](https://arxiv.org/abs/2406.16232)
- Tracy, K. & Manchester, Z. (2024). *On the Differentiability of the Primal-Dual
  Interior-Point Method*. arXiv:2406.11749. [qpax](https://github.com/kevin-tracy/qpax)
- TorchJD: [github.com/TorchJD/torchjd](https://github.com/TorchJD/torchjd)
