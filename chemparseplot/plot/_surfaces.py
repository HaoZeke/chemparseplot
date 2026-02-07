import jax
import jax.numpy as jnp
import jax.scipy.optimize as jopt
from jax import jit, vmap

# Force float32 for speed/viz
jax.config.update("jax_enable_x64", False)

# ==============================================================================
# 1. TPS IMPLEMENTATION (Legacy/Fallback)
# ==============================================================================


@jit
def _tps_solve(x, y, sm):
    # Kernel Matrix
    d2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)
    K = r**2 * jnp.log(r)
    K = K + jnp.eye(x.shape[0]) * sm

    # Polynomial Matrix
    N = x.shape[0]
    P = jnp.concatenate([jnp.ones((N, 1), dtype=jnp.float32), x], axis=1)
    M = P.shape[1]

    # Solve System
    zeros = jnp.zeros((M, M), dtype=jnp.float32)
    top = jnp.concatenate([K, P], axis=1)
    bot = jnp.concatenate([P.T, zeros], axis=1)
    lhs = jnp.concatenate([top, bot], axis=0)
    rhs = jnp.concatenate([y, jnp.zeros(M, dtype=jnp.float32)])

    coeffs = jnp.linalg.solve(lhs, rhs)
    return coeffs[:N], coeffs[N:]


# Standalone JIT-compiled predictor
@jit
def _tps_predict(x_query, x_obs, w, v):
    d2 = jnp.sum((x_query[:, None, :] - x_obs[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)
    K_q = r**2 * jnp.log(r)

    P_q = jnp.concatenate(
        [jnp.ones((x_query.shape[0], 1), dtype=jnp.float32), x_query], axis=1
    )
    return K_q @ w + P_q @ v


class FastTPS:
    def __init__(self, x_obs, y_obs, smoothing=1e-3, **kwargs):
        self.x_obs = jnp.asarray(x_obs, dtype=jnp.float32)
        self.y_obs = jnp.asarray(y_obs, dtype=jnp.float32)
        self.w, self.v = _tps_solve(self.x_obs, self.y_obs, smoothing)

    def __call__(self, x_query, chunk_size=500):
        """
        Batched prediction to prevent OOM errors on large grids.
        """
        x_query = jnp.asarray(x_query, dtype=jnp.float32)
        num_points = x_query.shape[0]

        preds = []
        # Process grid in small chunks to avoid exploding memory (MxNx2 tensor)
        for i in range(0, num_points, chunk_size):
            chunk = x_query[i : i + chunk_size]
            chunk_pred = _tps_predict(chunk, self.x_obs, self.w, self.v)
            preds.append(chunk_pred)

        return jnp.concatenate(preds, axis=0)


# ==============================================================================
# 2. STANDARD MATERN 5/2 (No Gradients)
# ==============================================================================


@jit
def _matern_solve(x, y, sm, length_scale):
    # Distance Matrix
    d2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)

    # Matérn 5/2 Kernel
    # k(r) = (1 + sqrt(5)r/l + 5r^2/3l^2) * exp(-sqrt(5)r/l)
    sqrt5_r_l = jnp.sqrt(5.0) * r / length_scale
    K = (1.0 + sqrt5_r_l + (5.0 * r**2) / (3.0 * length_scale**2)) * jnp.exp(-sqrt5_r_l)

    # Regularization (nugget)
    K = K + jnp.eye(x.shape[0]) * sm

    # 3. Solve (Cholesky is faster/stable for positive definite kernels like Matérn)
    # Note: We don't use the polynomial 'P' matrix here usually, as Matérn
    # decays to mean zero. If you want it to revert to a mean value, subtract
    # mean(y) before fitting and add it back after.
    L = jnp.linalg.cholesky(K)
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))

    return alpha, L


@jit
def _matern_predict(x_query, x_obs, alpha, length_scale):
    d2 = jnp.sum((x_query[:, None, :] - x_obs[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)

    sqrt5_r_l = jnp.sqrt(5.0) * r / length_scale
    K_q = (1.0 + sqrt5_r_l + (5.0 * r**2) / (3.0 * length_scale**2)) * jnp.exp(
        -sqrt5_r_l
    )

    return K_q @ alpha


class FastMatern:
    def __init__(self, x_obs, y_obs, smoothing=1e-3, length_scale=None, **kwargs):
        self.x_obs = jnp.asarray(x_obs, dtype=jnp.float32)
        self.y_obs = jnp.asarray(y_obs, dtype=jnp.float32)
        # Center the data (important for stationary kernels like Matérn)
        self.y_mean = jnp.mean(self.y_obs)
        y_centered = self.y_obs - self.y_mean

        # Auto-guess length_scale if not provided
        # Heuristic: Median distance of random subset to avoid O(N^2) just for valid param
        if length_scale is None:
            # Simple heuristic: sqrt(span) / 2 is often a safe start for density
            span = jnp.max(self.x_obs, axis=0) - jnp.min(self.x_obs, axis=0)
            self.length_scale = jnp.mean(span) * 0.2  # ~20% of the domain size
        else:
            self.length_scale = length_scale

        self.alpha, _ = _matern_solve(
            self.x_obs, y_centered, smoothing, self.length_scale
        )

    def __call__(self, x_query, chunk_size=500):
        """
        Batched prediction for Matern.
        """
        x_query = jnp.asarray(x_query, dtype=jnp.float32)
        num_points = x_query.shape[0]

        preds = []
        for i in range(0, num_points, chunk_size):
            chunk = x_query[i : i + chunk_size]
            chunk_pred = _matern_predict(
                chunk, self.x_obs, self.alpha, self.length_scale
            )
            preds.append(chunk_pred)

        return jnp.concatenate(preds, axis=0) + self.y_mean


# ==============================================================================
# 3. GRADIENT-ENHANCED MATERN (Optimizable & Batched)
# ==============================================================================


def matern_kernel(x1, x2, length_scale=1.0):
    d2 = jnp.sum((x1 - x2) ** 2)
    r = jnp.sqrt(d2 + 1e-12)
    ls = jnp.squeeze(length_scale)
    sqrt5_r_l = jnp.sqrt(5.0) * r / ls
    val = (1.0 + sqrt5_r_l + (5.0 * r**2) / (3.0 * ls**2)) * jnp.exp(-sqrt5_r_l)
    return val


# --- Auto-Diff the Kernel to get Gradient Covariances ---
# This creates a function that returns the (D+1)x(D+1) covariance block
# [ Cov(E, E)    Cov(E, dX)    Cov(E, dY) ]
# [ Cov(dX, E)   Cov(dX, dX)   Cov(dX, dY) ]
# [ Cov(dY, E)   Cov(dY, dX)   Cov(dY, dY) ]


def full_covariance_block(x1, x2, length_scale):
    # 0. Energy-Energy
    k_ee = matern_kernel(x1, x2, length_scale)

    # 1. Energy-Gradient (Jacobian of kernel w.r.t x2)
    k_ed = jax.grad(matern_kernel, argnums=1)(x1, x2, length_scale)

    # 2. Gradient-Energy (Jacobian of kernel w.r.t x1)
    k_de = jax.grad(matern_kernel, argnums=0)(x1, x2, length_scale)

    # 3. Gradient-Gradient (Hessian of kernel w.r.t x1, x2)
    # This captures how a slope at x1 correlates with a slope at x2
    k_dd = jax.jacfwd(jax.grad(matern_kernel, argnums=1), argnums=0)(
        x1, x2, length_scale
    )

    # Assemble the block:
    # Scalar k_ee, Vector k_ed, Vector k_de, Matrix k_dd

    # Top row: [E-E, E-dx, E-dy]
    row1 = jnp.concatenate([k_ee[None], k_ed])

    # Bottom rows: [dx-E,  dx-dx, dx-dy]
    #              [dy-E,  dy-dx, dy-dy]
    row2 = jnp.concatenate([k_de[:, None], k_dd], axis=1)

    block = jnp.concatenate([row1[None, :], row2], axis=0)
    return block


# Vectorize to create the full matrix
# maps over x1 (rows) and x2 (cols)
k_matrix_map = vmap(vmap(full_covariance_block, (None, 0, None)), (0, None, None))


# --- Cost Function (Optimization of Length Scale) ---
def negative_mll(log_params, x, y_flat, D_plus_1):
    """
    Optimizes length_scale and noise.
    """
    length_scale = jnp.exp(log_params[0])
    noise_scalar = jnp.exp(log_params[1])

    # Build Kernel Matrix
    K_blocks = k_matrix_map(x, x, length_scale)
    N = x.shape[0]
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)

    # Add Noise + Jitter
    # Optimization needs more jitter (1e-4) to stay positive definite
    diag_noise = (noise_scalar + 1e-4) * jnp.eye(N * D_plus_1)
    K_full = K_full + diag_noise

    # Factorize
    # Using eigh (eigenvalues) is slower but safer than Cholesky for stability checks,
    # but Cholesky is needed for speed. Crash/return Inf if it fails.
    L = jnp.linalg.cholesky(K_full)

    # Compute Terms
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y_flat))
    data_fit = 0.5 * jnp.dot(y_flat, alpha)
    complexity = jnp.sum(jnp.log(jnp.diag(L)))

    cost = data_fit + complexity

    # Check for NaNs and return infinite cost if bad
    return jnp.where(jnp.isnan(cost), 1e9, cost)


@jit
def _grad_matern_solve(x, y_full, noise_scalar, length_scale):
    K_blocks = k_matrix_map(x, x, length_scale)
    N, _, D_plus_1, _ = K_blocks.shape
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N * D_plus_1, N * D_plus_1)

    diag_noise = (noise_scalar + 1e-6) * jnp.eye(N * D_plus_1)
    K_full = K_full + diag_noise

    y_flat = y_full.flatten()
    # Robust solve for final step
    alpha = jnp.linalg.solve(K_full, y_flat)
    return alpha


@jit
def _grad_matern_predict(x_query, x_obs, alpha, length_scale):
    # This function builds the query matrix M x (N*(D+1))
    # If M is large, this explodes memory.
    # Must be called on chunks of x_query!
    def get_query_row(xq, xo):
        kee = matern_kernel(xq, xo, length_scale)
        ked = jax.grad(matern_kernel, argnums=1)(xq, xo, length_scale)
        return jnp.concatenate([kee[None], ked])

    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    M, N, D_plus_1 = K_q.shape
    K_q_flat = K_q.reshape(M, N * D_plus_1)
    return K_q_flat @ alpha


class GradientMatern:
    def __init__(
        self,
        x,
        y,
        gradients=None,
        smoothing=1e-4,
        length_scale=None,
        optimize=True,
        **kwargs,
    ):
        self.x = jnp.asarray(x, dtype=jnp.float32)

        # Prepare targets
        y_energies = jnp.asarray(y, dtype=jnp.float32)[:, None]

        if gradients is not None:
            grad_vals = jnp.asarray(gradients, dtype=jnp.float32)
        else:
            grad_vals = jnp.zeros_like(self.x)

        self.y_full = jnp.concatenate([y_energies, grad_vals], axis=1)

        # Center the energy mean (gradients have 0 mean typically)
        self.e_mean = jnp.mean(y_energies)
        self.y_full = self.y_full.at[:, 0].add(-self.e_mean)
        self.y_flat = self.y_full.flatten()

        D_plus_1 = self.x.shape[1] + 1
        self.smoothing = smoothing

        # --- Heuristics ---
        if length_scale is None:
            span = jnp.max(self.x, axis=0) - jnp.min(self.x, axis=0)
            init_ls = jnp.mean(span) * 0.5
        else:
            init_ls = length_scale

        # Clamp initial noise to prevent log(0)
        init_noise = max(smoothing, 1e-4)

        if optimize:
            # Optimize [log_ls, log_noise]
            x0 = jnp.array([jnp.log(init_ls), jnp.log(init_noise)])

            def loss_fn(log_p):
                return negative_mll(log_p, self.x, self.y_flat, D_plus_1)

            results = jopt.minimize(loss_fn, x0, method="BFGS", tol=1e-3)

            # Unpack results safely
            learned_ls = float(jnp.exp(results.x[0]))
            learned_noise = float(jnp.exp(results.x[1]))

            # Check for NaNs (optimization failure)
            if jnp.isnan(learned_ls) or jnp.isnan(learned_noise):
                print("Warning: Optimization failed (NaN). Reverting to heuristics.")
                self.ls = init_ls
                self.noise = init_noise
            else:
                self.ls = learned_ls
                self.noise = learned_noise
        else:
            self.ls = init_ls
            self.noise = init_noise

        self.alpha = _grad_matern_solve(self.x, self.y_full, self.noise, self.ls)

    def __call__(self, x_query, chunk_size=500):
        """
        Batched prediction to prevent OOM errors on large grids.
        """
        x_query = jnp.asarray(x_query, dtype=jnp.float32)
        num_points = x_query.shape[0]

        preds = []
        for i in range(0, num_points, chunk_size):
            chunk = x_query[i : i + chunk_size]
            chunk_pred = _grad_matern_predict(chunk, self.x, self.alpha, self.ls)
            preds.append(chunk_pred)

        return jnp.concatenate(preds, axis=0) + self.e_mean


# Factory for string-based instantiation
def get_surface_model(name):
    if name == "grad_matern":
        return GradientMatern
    if name == "tps":
        return FastTPS
    if name == "matern":
        return FastMatern
    if name == "rbf":
        return FastTPS  # Legacy default
    raise ValueError(f"Unknown surface model: {name}")
