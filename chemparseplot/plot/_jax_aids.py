import jax
import jax.numpy as jnp
from jax import jit

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
