# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Gold adatom hopping on Al(100) with EMT: NEB histories, bridge saddle, adatom map.

Generates the data behind the figure tests and the documentation figures.
System and NEB set-up are those of the ASE tutorial "Diffusion of gold atom on
Al(100) surface (NEB)": a 2x2x3 Al(100) slab, Au 1.7 A above the hollow site,
4 A of vacuum, the two lower layers fixed, three intermediate images, linear
interpolation, the ASE default spring constant and fmax = 0.05 eV/A.

- NEB: run once with BFGS and once with FIRE, recording every iteration. The
  band uses the improved tangent of Henkelman and Jonsson, J. Chem. Phys. 113,
  9978 (2000), doi:10.1063/1.1323224 ("improvedtangent", Paper I of the ASE
  NEB docstring). It is named because ASE's default changed: up to 3.27 it was
  ASE's own unpublished "aseneb" tangent, from 3.28 it is the improved one.
- Reference saddle: the bridge site. The surface has a mirror plane through the
  bridge, normal to the hop, that swaps the two hollows, so the in-plane force
  on Au vanishes there by symmetry; relaxing Au's height and the top layer with
  Au's x, y fixed at the bridge gives the saddle. It is relaxed to
  fmax = sqrt(eps) eV/A, which puts its energy error at the double-precision
  floor.
- Map: one periodic cell (2 a_s x 2 a_s) of Au positions, each relaxed with
  Au's x, y fixed, to the reference fmax, so the relaxation error is far
  below the interpolation error being measured (at the NEB fmax it puts a
  floor of a few meV under it). Spacings a_s / 2^k keep hollows, bridges and
  top sites on nodes. The spacing is halved until the bilinear
  interpolation error, measured by also relaxing at every cell centre, is
  below half an entry of a matplotlib colour map spread over the map's
  energy range, so a figure drawn from the nodes shows what the exact
  surface would.

Usage: python au_al100.py FULL.json FIXTURE.json

FULL.json holds everything, including all image positions and every map
level; FIXTURE.json is the subset the tests and the documentation figures
read (no positions, only the map level that met the tolerance).
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import ase
import matplotlib as mpl
import numpy as np
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixCartesian
from ase.mep import NEB
from ase.optimize import BFGS, FIRE, QuasiNewton

FMAX = 0.05
NEB_METHOD = "improvedtangent"
N_INTERMEDIATE = 3
ADSORPTION_HEIGHT = 1.7
VACUUM = 4.0
FMAX_REFERENCE = float(np.sqrt(np.finfo(float).eps))
COLORMAP_ENTRIES = mpl.rcParamsDefault["image.lut"]


def fixed_layers(atoms):
    return FixAtoms(mask=[atom.tag > 1 for atom in atoms])


def with_emt(atoms, constraints=None):
    atoms = atoms.copy()
    atoms.set_constraint(constraints if constraints is not None else fixed_layers(atoms))
    atoms.calc = EMT()
    return atoms


def endpoints():
    slab = fcc100("Al", size=(2, 2, 3))
    add_adsorbate(slab, "Au", ADSORPTION_HEIGHT, "hollow")
    slab.center(axis=2, vacuum=VACUUM)
    slab = with_emt(slab)
    QuasiNewton(slab, logfile=None).run(fmax=FMAX)
    initial = with_emt(slab)
    slab[-1].x += slab.get_cell()[0, 0] / 2
    QuasiNewton(slab, logfile=None).run(fmax=FMAX)
    final = with_emt(slab)
    return initial, final


def path_coordinates(pos, forces):
    """Cumulative image distance and the true force along the band tangent."""
    flat = pos.reshape(len(pos), -1)
    seg = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tangents = np.empty_like(flat)
    tangents[0] = flat[1] - flat[0]
    tangents[-1] = flat[-1] - flat[-2]
    tangents[1:-1] = flat[2:] - flat[:-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    f_para = np.einsum("ij,ij->i", forces.reshape(len(pos), -1), tangents)
    return s, f_para


def run_neb(optimizer, initial, final):
    images = [with_emt(initial)]
    images += [with_emt(initial) for _ in range(N_INTERMEDIATE)]
    images.append(with_emt(final))
    neb = NEB(images, method=NEB_METHOD)
    neb.interpolate()
    rows = []

    def record():
        pos = np.array([im.get_positions() for im in images])
        energies = np.array([im.get_potential_energy() for im in images])
        forces = np.array([im.get_forces() for im in images])
        band = neb.get_forces()
        fmax = float(np.sqrt((band**2).sum(axis=1).max()))
        step = 0.0
        if rows:
            prev = np.array(rows[-1]["positions"])
            step = float(np.linalg.norm(pos - prev, axis=2).max())
        s, f_para = path_coordinates(pos, forces)
        rows.append(
            {
                "iteration": len(rows),
                "fmax": fmax,
                "step": step,
                "energies": energies.tolist(),
                "s": s.tolist(),
                "f_para": f_para.tolist(),
                "au_xy": pos[:, -1, :2].tolist(),
                "positions": pos.tolist(),
            }
        )

    opt = optimizer(neb, logfile=None)
    opt.attach(record, interval=1)
    t0 = time.perf_counter()
    converged = opt.run(fmax=FMAX)
    wall = time.perf_counter() - t0
    return {"converged": bool(converged), "wall_s": wall, "history": rows}


def bridge_saddle(initial, final):
    au = len(initial) - 1
    guess = initial.copy()
    guess.positions[:] = 0.5 * (initial.positions + final.positions)
    constraints = [fixed_layers(guess), FixCartesian(au, mask=(True, True, False))]
    sp = with_emt(guess, constraints)
    xy0 = sp.positions[au, :2].copy()
    t0 = time.perf_counter()
    BFGS(sp, logfile=None).run(fmax=FMAX_REFERENCE)
    wall = time.perf_counter() - t0
    free = with_emt(sp)
    in_plane = free.get_forces()[au, :2]
    return {
        "energy": float(sp.get_potential_energy()),
        "au_xyz": sp.positions[au].tolist(),
        "au_xy_moved": float(np.abs(sp.positions[au, :2] - xy0).max()),
        "free_force_au_xy": in_plane.tolist(),
        "wall_s": wall,
    }


_TEMPLATE = None


def _init_worker(template):
    global _TEMPLATE  # noqa: PLW0603
    _TEMPLATE = template


def _relaxed_at(xy):
    atoms = _TEMPLATE.copy()
    au = len(atoms) - 1
    atoms.positions[au, :2] = xy
    constraints = [fixed_layers(atoms), FixCartesian(au, mask=(True, True, False))]
    atoms = with_emt(atoms, constraints)
    BFGS(atoms, logfile=None).run(fmax=FMAX_REFERENCE)
    return float(atoms.get_potential_energy())


def _relax_all(pool, xs, ys):
    pts = [(x, y) for y in ys for x in xs]
    return np.array(list(pool.map(_relaxed_at, pts))).reshape(len(ys), len(xs))


def adatom_map(initial, final):
    au = len(initial) - 1
    a_s = initial.get_cell()[0, 0] / 2
    bridge = 0.5 * (initial.positions[au, :2] + final.positions[au, :2])
    levels = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        os.cpu_count(), initializer=_init_worker, initargs=(initial,)
    ) as pool:
        nodes_per_half = 1
        while True:
            t_level = time.perf_counter()
            h = a_s / (2 * nodes_per_half)
            n = 4 * nodes_per_half + 1
            x = bridge[0] + h * (np.arange(n) - (n - 1) / 2)
            y = bridge[1] + h * (np.arange(n) - (n - 1) / 2)
            grid = _relax_all(pool, x, y)
            centres = _relax_all(pool, x[:-1] + h / 2, y[:-1] + h / 2)
            bilinear = 0.25 * (
                grid[:-1, :-1] + grid[1:, :-1] + grid[:-1, 1:] + grid[1:, 1:]
            )
            err = float(np.abs(centres - bilinear).max())
            tol = float(np.ptp(grid)) / (2 * COLORMAP_ENTRIES)
            levels.append(
                {
                    "nodes_per_half_spacing": nodes_per_half,
                    "spacing": float(h),
                    "centre_interpolation_error_max": err,
                    "tolerance": tol,
                    "relaxations": int(grid.size + centres.size),
                    "wall_s": time.perf_counter() - t_level,
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "energy": grid.tolist(),
                }
            )
            print(
                f"map a_s/{2 * nodes_per_half}: {n}x{n} nodes, max centre error"
                f" {err * 1e3:.3f} meV, tolerance {tol * 1e3:.3f} meV",
                flush=True,
            )
            if err <= tol:
                break
            nodes_per_half *= 2
    final_level = levels[-1]
    return {
        "a_s": float(a_s),
        "levels": [
            {k: v for k, v in lv.items() if k not in ("x", "y", "energy")}
            for lv in levels
        ],
        "by_nodes_per_half_spacing": {
            str(lv["nodes_per_half_spacing"]): lv for lv in levels
        },
        "spacing": final_level["spacing"],
        "x": final_level["x"],
        "y": final_level["y"],
        "energy": final_level["energy"],
        "centre_interpolation_error_max": final_level["centre_interpolation_error_max"],
        "tolerance": final_level["tolerance"],
        "wall_s": time.perf_counter() - t0,
        "relaxations": int(sum(lv["relaxations"] for lv in levels)),
    }


def hardware():
    model = "unrecorded"
    lscpu = shutil.which("lscpu")
    if lscpu is not None:
        out = subprocess.run([lscpu], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name:"):
                model = line.split(":", 1)[1].strip()
    return {"cpu": model, "logical_cpus": os.cpu_count(), "machine": platform.machine()}


def fixture(full):
    """The subset the tests read: no image positions, only the converged map."""
    out = {k: v for k, v in full.items() if k not in ("neb", "map")}
    out["neb"] = {
        name: {
            **{k: v for k, v in run.items() if k != "history"},
            "history": [
                {k: v for k, v in r.items() if k != "positions"} for r in run["history"]
            ],
        }
        for name, run in full["neb"].items()
    }
    out["map"] = {
        k: v for k, v in full["map"].items() if k != "by_nodes_per_half_spacing"
    }
    return out


def main(full_path, fixture_path):
    t0 = time.perf_counter()
    initial, final = endpoints()
    e0 = float(initial.get_potential_energy())
    out = {
        "meta": {
            "system": "Au adatom on Al(100), 2x2x3 slab, EMT (ASE tutorial set-up)",
            "fmax": FMAX,
            "fmax_reference": FMAX_REFERENCE,
            "neb_method": NEB_METHOD,
            "n_intermediate": N_INTERMEDIATE,
            "spring_constant": inspect.signature(NEB.__init__).parameters["k"].default,
            "colormap_entries": COLORMAP_ENTRIES,
            "python": sys.version.split()[0],
            "ase": ase.__version__,
            "numpy": np.__version__,
            "hardware": hardware(),
        },
        "e_initial": e0,
        "e_final": float(final.get_potential_energy()),
        "au_initial_xyz": initial.positions[-1].tolist(),
        "au_final_xyz": final.positions[-1].tolist(),
        "neb": {
            name: run_neb(opt, initial, final)
            for name, opt in (("BFGS", BFGS), ("FIRE", FIRE))
        },
        "saddle": bridge_saddle(initial, final),
        "map": adatom_map(initial, final),
    }
    out["meta"]["wall_s_total"] = time.perf_counter() - t0
    with open(full_path, "w") as fh:
        json.dump(out, fh)
    with open(fixture_path, "w") as fh:
        json.dump(fixture(out), fh, indent=0)
        fh.write("\n")
    sp = out["saddle"]
    saddle_barrier = sp["energy"] - e0
    in_plane = np.abs(sp["free_force_au_xy"]).max()
    print(f"E_initial = {e0:.10f} eV, E_final - E_initial = {out['e_final'] - e0:.3e} eV")
    print(
        f"bridge barrier = {saddle_barrier:.10f} eV;"
        f" Au xy moved {sp['au_xy_moved']:.1e} A;"
        f" free in-plane force on Au {in_plane:.2e} eV/A"
    )
    for name, run in out["neb"].items():
        last = run["history"][-1]
        barrier = max(last["energies"]) - e0
        print(
            f"{name}: converged={run['converged']} iterations={last['iteration']}"
            f" fmax={last['fmax']:.4f} barrier={barrier:.10f} eV"
            f" wall={run['wall_s']:.2f} s"
        )
    m = out["map"]
    err, tol = m["centre_interpolation_error_max"], m["tolerance"]
    print(
        f"map: {len(m['x'])}x{len(m['y'])} nodes, spacing {m['spacing']:.4f} A,"
        f" {m['relaxations']} relaxations in {m['wall_s']:.1f} s,"
        f" max centre interpolation error {err * 1e3:.3f} meV"
        f" (tolerance {tol * 1e3:.3f} meV)"
    )
    print(f"total wall {out['meta']['wall_s_total']:.1f} s on {out['meta']['hardware']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
