import jax
import jax.numpy as jnp
from jax import jit, vmap

jax.config.update("jax_enable_x64", False)


# Standalone JIT-compiled solver (functional core)
@jit
def _tps_solve(x, y, sm):
    # 1. Kernel Matrix
    d2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)
    K = r**2 * jnp.log(r)
    K = K + jnp.eye(x.shape[0]) * sm

    # 2. Polynomial Matrix
    N = x.shape[0]
    P = jnp.concatenate([jnp.ones((N, 1), dtype=jnp.float32), x], axis=1)
    M = P.shape[1]

    # 3. Solve System
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
    def __init__(self, x_obs, y_obs, smoothing=1e-3):
        self.x_obs = jnp.asarray(x_obs, dtype=jnp.float32)
        self.y_obs = jnp.asarray(y_obs, dtype=jnp.float32)
        # Call the JIT-ed solver
        self.w, self.v = _tps_solve(self.x_obs, self.y_obs, smoothing)

    def __call__(self, x_query):
        x_query = jnp.asarray(x_query, dtype=jnp.float32)
        # Call the JIT-ed predictor
        return _tps_predict(x_query, self.x_obs, self.w, self.v)


# JIT-compiled Matérn 5/2 Solver
@jit
def _matern_solve(x, y, sm, length_scale):
    # 1. Distance Matrix
    d2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(d2 + 1e-12)

    # 2. Matérn 5/2 Kernel
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
    def __init__(self, x_obs, y_obs, smoothing=1e-3, length_scale=None):
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

    def __call__(self, x_query):
        x_query = jnp.asarray(x_query, dtype=jnp.float32)
        pred_centered = _matern_predict(
            x_query, self.x_obs, self.alpha, self.length_scale
        )
        return pred_centered + self.y_mean
