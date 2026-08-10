# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Sketch-map landscape figures over minima databases.

A campaign of quenched minima becomes one figure: a sketch-map plane
(lab-cosmo ``dimlandmark``/``dimred`` binaries) over a permutation-invariant
descriptor, a filled-contour energy surface with thin contour lines, per-arm
scatter overlays, pinned reference markers, and rendered structure insets in
the figure margins tied to their map points.

The default descriptor is the sorted smoothed-coordination-number vector,
the discriminant of the sketch-map literature for Lennard-Jones clusters:
it separates the fcc funnel from the icosahedral one. SOAP power spectra
(through ``featomic``) remain available for molecular systems where
coordination alone degenerates.

```{versionadded} 1.10.0
```
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from chemparseplot.plot.theme import RUHI_THEME

ARM_MARKERS = ["o", "^", "s", "P", "X"]

SOAP_HYPERS = {
    "cutoff": {"radius": 4.0, "smoothing": {"type": "ShiftedCosine", "width": 0.5}},
    "density": {"type": "Gaussian", "width": 0.35},
    "basis": {
        "type": "TensorProduct",
        "max_angular": 4,
        "radial": {"type": "Gto", "max_radial": 5},
    },
}


def cn_matrix(structures, symbols, cutoff=1.5, cn_species=None):
    """Sorted smoothed coordination numbers, permutation invariant.

    A Fermi switching function at the cutoff keeps the vector continuous,
    so near-degenerate minima do not collapse onto identical rows.

    Parameters
    ----------
    structures : sequence of array-like
        Flat ``3N`` coordinate rows.
    symbols : list of str
        One chemical symbol per atom.
    cutoff : float
        Coordination cutoff in the coordinate units.
    cn_species : str, optional
        Restrict the descriptor to atoms of one species.

    ```{versionadded} 1.10.0
    ```
    """
    width = 0.1 * cutoff
    keep = None
    if cn_species:
        keep = np.array([s == cn_species for s in symbols])
    rows = []
    for x in structures:
        pos = np.asarray(x, dtype=float).reshape(-1, 3)
        if keep is not None:
            pos = pos[keep]
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        cn = (1.0 / (1.0 + np.exp((d - cutoff) / width))).sum(axis=1)
        rows.append(np.sort(cn))
    return np.vstack(rows)


def soap_matrix(structures, symbols, cutoff=4.0, hypers=None):
    """Structure-averaged SOAP power spectra through featomic.

    ```{versionadded} 1.10.0
    ```
    """
    import ase
    from featomic import SoapPowerSpectrum

    hy = dict(hypers or SOAP_HYPERS)
    hy["cutoff"] = dict(hy["cutoff"], radius=cutoff)
    calc = SoapPowerSpectrum(**hy)
    rows = []
    for x in structures:
        pos = np.asarray(x, dtype=float).reshape(-1, 3)
        pos = pos - pos.mean(axis=0)
        atoms = ase.Atoms(symbols=symbols, positions=pos)
        ps = calc.compute(atoms)
        ps = ps.keys_to_samples("center_type").keys_to_properties(
            ["neighbor_1_type", "neighbor_2_type"]
        )
        rows.append(ps.block().values.mean(axis=0))
    return np.vstack(rows)


def median_sigma(matrix, rng, sample=400):
    """The sketch-map switching scale: median pairwise descriptor distance.

    ```{versionadded} 1.10.0
    ```
    """
    sub = matrix[rng.choice(len(matrix), size=min(len(matrix), sample), replace=False)]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    return float(np.median(d[np.triu_indices(len(sub), 1)]))


def run_sketchmap(
    matrix,
    smbin,
    lapack_lib,
    n_landmark,
    sigma,
    workdir,
    n_pinned=0,
):
    """Landmark selection and the published sketch-map projection schedule.

    ``dimlandmark`` picks minmax landmarks (the first ``n_pinned`` rows are
    pinned); ``dimred`` runs sigmoid exponents 8,8 against 2,8, conjugate
    gradient pre-optimization, then the pointwise global grid stage. No
    ``-pi``: descriptor space is not periodic.

    Returns
    -------
    coords, idx : ndarray
        Projected landmark coordinates and their row indices in ``matrix``.

    ```{versionadded} 1.10.0
    ```
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    smbin = Path(smbin)
    mat = workdir / "desc.dat"
    np.savetxt(mat, matrix, fmt="%.8f")
    env = {"LD_LIBRARY_PATH": str(lapack_lib), "PATH": "/usr/bin:/bin"}
    dim = matrix.shape[1]
    lm_raw = subprocess.run(  # noqa: S603
        [
            str(smbin / "dimlandmark"),
            "-D",
            str(dim),
            "-n",
            str(n_landmark),
            "-mode",
            "minmax",
            "-i",
            "-ifirst",
            str(n_pinned),
        ],
        stdin=mat.open(),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    rows = [
        line.split()
        for line in lm_raw.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    idx = [int(r[0]) for r in rows]
    # Guarantee the pinned rows: dimlandmark's -ifirst only seeds the greedy
    # walk, and farthest-point sampling skips anything near an already-kept
    # point, which is exactly where a pinned reference's twin sits.
    selected = [i for i in idx if i >= n_pinned]
    keep = list(range(n_pinned)) + selected
    keep = keep[:max(n_landmark, n_pinned)]
    idx = np.array(keep)
    (workdir / "landmarks.dat").write_text(
        "\n".join(
            " ".join(f"{v:.8f}" for v in matrix[i]) for i in keep
        )
        + "\n"
    )
    proj = subprocess.run(  # noqa: S603
        [
            str(smbin / "dimred"),
            "-D",
            str(dim),
            "-d",
            "2",
            "-center",
            "-fun-hd",
            f"{sigma},8,8",
            "-fun-ld",
            f"{sigma},2,8",
            "-preopt",
            "500",
            "-grid",
            f"{10 * sigma:.4f},51,501",
            "-gopt",
            "10",
        ],
        stdin=(workdir / "landmarks.dat").open(),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    coords = np.array(
        [
            [float(v) for v in line.split()[:2]]
            for line in proj.splitlines()
            if line.strip() and not line.startswith("#")
        ]
    )
    if len(idx) != len(coords):
        msg = (
            f"dimlandmark gave {len(idx)} indices, dimred {len(coords)} "
            "points; refusing to plot with mismatched labels"
        )
        raise RuntimeError(msg)
    return coords, idx


def plot_landscape_surface(ax, coords, energies, cmap=None, levels=20):
    """Filled-contour energy surface with thin contour lines.

    Returns the contour set for colorbar wiring.

    ```{versionadded} 1.10.0
    ```
    """
    from scipy.interpolate import griddata

    cmap = cmap or mpl.colormaps["ruhi_diverging"]
    vmin = float(np.nanmin(energies))
    vmax = float(np.nanpercentile(energies, 97))
    gx, gy = np.meshgrid(
        np.linspace(coords[:, 0].min(), coords[:, 0].max(), 240),
        np.linspace(coords[:, 1].min(), coords[:, 1].max(), 240),
    )
    surface = griddata(
        coords, np.clip(energies, vmin, vmax), (gx, gy), method="linear"
    )
    lv = np.linspace(vmin, vmax, levels)
    cf = ax.contourf(gx, gy, surface, levels=lv, cmap=cmap, vmin=vmin, vmax=vmax,
                     alpha=0.85)
    ax.contour(gx, gy, surface, levels=lv, colors="black", linewidths=0.3,
               alpha=0.45)
    ax.set_facecolor(RUHI_THEME.gridcolor)
    return cf


def margin_inset_slots(n):
    """Axes-fraction slots down the left and right figure margins.

    ```{versionadded} 1.10.0
    ```
    """
    slots = [(-0.26, 0.82), (-0.26, 0.22), (1.30, 0.82), (1.30, 0.22),
             (-0.26, 0.52), (1.30, 0.52)]
    return slots[:n]


def place_margin_insets(fig, ax, entries, zoom=0.14, renderer="xyzrender"):
    """Structure insets at margin slots, arrowed to their map points.

    Parameters
    ----------
    entries : list of dict
        Each carries ``atoms`` (ASE Atoms), ``xy`` (map point), and
        optionally ``label`` drawn under the slot.

    ```{versionadded} 1.10.0
    ```
    """
    from chemparseplot.plot.neb import plot_structure_inset

    slots = margin_inset_slots(len(entries))
    fig.canvas.draw()
    remaining = list(range(len(slots)))
    for entry in entries:
        x, y = entry["xy"]
        pt_disp = ax.transData.transform((x, y))
        if not remaining:
            break
        best = min(
            remaining,
            key=lambda s: np.linalg.norm(ax.transAxes.transform(slots[s]) - pt_disp),
        )
        remaining.remove(best)
        slot_disp = ax.transAxes.transform(slots[best])
        offset_pts = (slot_disp - pt_disp) * 72.0 / fig.dpi
        plot_structure_inset(
            ax,
            entry["atoms"],
            x,
            y,
            xybox=tuple(offset_pts),
            rad=0.08,
            zoom=zoom,
            rotation="auto",
            renderer=renderer,
            xyzrender_config="paton",
        )
        ax.scatter([x], [y], marker="o", s=120, facecolors="none",
                   edgecolors="black", linewidths=1.4, zorder=7)
        if entry.get("label"):
            ax.annotate(
                entry["label"],
                xy=slots[best],
                xycoords="axes fraction",
                xytext=(0, -58),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=plt.rcParams["font.size"] - 3,
                color="black",
                weight="bold",
                zorder=81,
                annotation_clip=False,
            )
