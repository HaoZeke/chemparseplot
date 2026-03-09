"""Generate minimal synthetic HDF5 fixtures for ChemGP tutorials.

Run once to create the data/ directory contents:
    python generate_sample_data.py

Not committed to git; the generated .h5 files are committed instead.
"""

import numpy as np
import h5py
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)


def _write_convergence(path: Path):
    """3 methods x 15 steps convergence data."""
    with h5py.File(path, "w") as f:
        f.attrs["conv_tol"] = 0.5

        g = f.create_group("table/convergence")

        methods = []
        calls = []
        forces = []

        for method, rate in [("GP-NEB", 0.7), ("AIE", 0.75), ("OIE", 0.8)]:
            n = 15
            c = np.arange(1, n + 1) * 10
            force = 5.0 * np.exp(-rate * np.arange(n)) + RNG.normal(0, 0.05, n)
            force = np.maximum(force, 0.01)
            methods.extend([method] * n)
            calls.extend(c.tolist())
            forces.extend(force.tolist())

        g.create_dataset("oracle_calls", data=np.array(calls))
        g.create_dataset("max_fatom", data=np.array(forces))
        # Store method as variable-length strings
        dt = h5py.string_dtype()
        g.create_dataset("method", data=np.array(methods, dtype=object), dtype=dt)


def _write_surface(path: Path):
    """40x40 grid + 1 path + 2 point sets."""
    n = 40
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)
    # Muller-Brown-like double well
    Z = (X**2 - 1) ** 2 + 2 * Y**2 + 0.5 * X * Y

    with h5py.File(path, "w") as f:
        # Energy grid
        eg = f.create_dataset("grids/energy", data=Z)
        eg.attrs["x_range"] = [-2.0, 2.0]
        eg.attrs["y_range"] = [-2.0, 2.0]
        eg.attrs["x_length"] = n
        eg.attrs["y_length"] = n

        # Variance grid
        V = np.exp(-X**2 - Y**2) * 2.0
        vg = f.create_dataset("grids/variance", data=V)
        vg.attrs["x_range"] = [-2.0, 2.0]
        vg.attrs["y_range"] = [-2.0, 2.0]
        vg.attrs["x_length"] = n
        vg.attrs["y_length"] = n

        # GP progression grids
        for n_train in [5, 10, 20]:
            noise = RNG.normal(0, 3.0 / n_train, Z.shape)
            gp_mean = Z + noise
            gm = f.create_dataset(f"grids/gp_mean_{n_train}", data=gp_mean)
            gm.attrs["x_range"] = [-2.0, 2.0]
            gm.attrs["y_range"] = [-2.0, 2.0]
            gm.attrs["x_length"] = n
            gm.attrs["y_length"] = n

            tx = RNG.uniform(-2, 2, n_train)
            ty = RNG.uniform(-2, 2, n_train)
            pg = f.create_group(f"points/training_{n_train}")
            pg.create_dataset("x", data=tx)
            pg.create_dataset("y", data=ty)

        # NEB path
        t = np.linspace(0, np.pi, 7)
        px = -1 + 2 * t / np.pi
        py = 0.3 * np.sin(t)
        p = f.create_group("paths/neb")
        p.create_dataset("x", data=px)
        p.create_dataset("y", data=py)

        # Stationary points
        mins = f.create_group("points/minima")
        mins.create_dataset("x", data=np.array([-1.0, 1.0]))
        mins.create_dataset("y", data=np.array([0.0, 0.0]))

        sads = f.create_group("points/saddles")
        sads.create_dataset("x", data=np.array([0.0]))
        sads.create_dataset("y", data=np.array([0.0]))

        # Training points
        train = f.create_group("points/training")
        train.create_dataset("x", data=RNG.uniform(-2, 2, 12))
        train.create_dataset("y", data=RNG.uniform(-2, 2, 12))


def _write_nll(path: Path):
    """25x25 NLL grid + optimum point."""
    n = 25
    ls = np.linspace(-3, 3, n)
    sv = np.linspace(-3, 3, n)
    LS, SV = np.meshgrid(ls, sv)
    # Quadratic bowl with minimum near (0.5, -1)
    NLL = 2.0 * (LS - 0.5) ** 2 + 3.0 * (SV + 1.0) ** 2 + 10.0

    with h5py.File(path, "w") as f:
        ng = f.create_dataset("grids/nll", data=NLL)
        ng.attrs["x_range"] = [-3.0, 3.0]
        ng.attrs["y_range"] = [-3.0, 3.0]
        ng.attrs["x_length"] = n
        ng.attrs["y_length"] = n

        # Gradient norm
        grad_ls = 4.0 * (LS - 0.5)
        grad_sv = 6.0 * (SV + 1.0)
        grad_norm = np.sqrt(grad_ls**2 + grad_sv**2)
        gg = f.create_dataset("grids/gradient_norm", data=grad_norm)
        gg.attrs["x_range"] = [-3.0, 3.0]
        gg.attrs["y_range"] = [-3.0, 3.0]
        gg.attrs["x_length"] = n
        gg.attrs["y_length"] = n

        f.attrs["log_sigma2"] = 0.5
        f.attrs["log_theta"] = -1.0


if __name__ == "__main__":
    _write_convergence(DATA / "sample_convergence.h5")
    _write_surface(DATA / "sample_surface.h5")
    _write_nll(DATA / "sample_nll.h5")
    print(f"Generated 3 fixtures in {DATA}/")
