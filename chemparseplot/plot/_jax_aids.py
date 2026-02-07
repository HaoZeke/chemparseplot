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

# --- The Core Kernel (Scalar) ---
def matern_kernel(x1, x2, length_scale=1.0):
    d2 = jnp.sum((x1 - x2)**2)
    r = jnp.sqrt(d2 + 1e-12)
    sqrt5_r_l = jnp.sqrt(5.0) * r / length_scale
    val = (1.0 + sqrt5_r_l + (5.0 * r**2) / (3.0 * length_scale**2)) * jnp.exp(-sqrt5_r_l)
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
    k_dd = jax.jacfwd(jax.grad(matern_kernel, argnums=1), argnums=0)(x1, x2, length_scale)
    
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

@jit
def _grad_matern_solve(x, y_full, sm, length_scale):
    # x: (N, D)
    # y_full: (N, D+1) -> flattens to (N*(D+1))
    
    # Build huge block matrix
    # shape (N, N, D+1, D+1)
    K_blocks = k_matrix_map(x, x, length_scale)
    
    N, _, D_plus_1, _ = K_blocks.shape
    
    # Reshape to 2D matrix (N*(D+1), N*(D+1))
    K_full = K_blocks.transpose(0, 2, 1, 3).reshape(N*D_plus_1, N*D_plus_1)
    
    # Regularization (nugget)
    # We might want less noise on energies than gradients, but uniform is fine for now
    K_full = K_full + jnp.eye(N*D_plus_1) * sm
    
    # Flatten Targets
    y_flat = y_full.flatten() # [E1, dx1, dy1, E2, dx2, dy2...]
    
    # Solve
    L = jnp.linalg.cholesky(K_full)
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y_flat))
    
    return alpha

@jit
def _grad_matern_predict(x_query, x_obs, alpha, length_scale):
    # We only want Energy predictions (top row of the covariance blocks)
    # But we depend on the full alpha vector (which includes gradient weights)
    
    # Get Energy-Energy and Energy-Gradient parts
    # k_query_blocks shape: (M, N, 1, D+1)  <-- we only need the first row of the block
    
    def get_query_row(xq, xo):
        # Cov(E_query, E_obs)
        kee = matern_kernel(xq, xo, length_scale)
        # Cov(E_query, Grad_obs)
        ked = jax.grad(matern_kernel, argnums=1)(xq, xo, length_scale)
        return jnp.concatenate([kee[None], ked])

    # (M, N, D+1)
    K_q = vmap(vmap(get_query_row, (None, 0)), (0, None))(x_query, x_obs)
    
    # Flatten last two dims to match alpha: (M, N*(D+1))
    M, N, D_plus_1 = K_q.shape
    K_q_flat = K_q.reshape(M, N*D_plus_1)
    
    return K_q_flat @ alpha

class GradientMatern:
    def __init__(self, x, y, gradients=None, smoothing=1e-4, length_scale=1.0, **kwargs):
        """
        x: (N, 2) array of coordinates
        y: (N,) array of energies
        gradients: (N, 2) array of forces (-grad V). 
                   Note: The kernel expects gradients of the function. 
                   If passing forces, flip sign if y is Potential Energy.
        """
        self.x = jnp.asarray(x, dtype=jnp.float32)
        
        # Prepare targets: [Energy, dE/dx, dE/dy] for each point
        # Shape becomes (N, 3)
        y_energies = jnp.asarray(y, dtype=jnp.float32)[:, None]
        
        if gradients is not None:
            # Assumes gradients are passed as dE/dx (e.g. -Force)
            grad_vals = jnp.asarray(gradients, dtype=jnp.float32)
        else:
            grad_vals = jnp.zeros_like(self.x)
            
        self.y_full = jnp.concatenate([y_energies, grad_vals], axis=1)
        
        # Center the energy mean (gradients have 0 mean typically)
        self.e_mean = jnp.mean(y_energies)
        self.y_full = self.y_full.at[:, 0].add(-self.e_mean)
        
        self.ls = length_scale
        self.alpha = _grad_matern_solve(self.x, self.y_full, smoothing, self.ls)

    def __call__(self, x_query):
        x_q = jnp.asarray(x_query, dtype=jnp.float32)
        return _grad_matern_predict(x_q, self.x, self.alpha, self.ls) + self.e_mean
