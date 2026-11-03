# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Structure figures for clusters, periodic cells, and reaction paths.

The backend is fixed by the kind of system. A Lennard-Jones cluster is drawn
with solvis. A cell, a reaction path, and any snapshot that contains more
than one element are drawn with xyzrender. A one-element snapshot uses the
cluster backend, with Cordero radii instead of the Lennard-Jones contact
radius.

```{versionadded} 1.14.0
```
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chemparseplot.plot.theme import RUHI_COLORS, RUHI_THEME, PlotTheme

log = logging.getLogger(__name__)

# Nature figure sizing: a single column is 89 mm, and photographs are
# requested at 300 dpi. 25.4 mm is one inch.
NATURE_SINGLE_COLUMN_MM = 89.0
PRINT_DPI = 300
MM_PER_INCH = 25.4
CANVAS = round(NATURE_SINGLE_COLUMN_MM / MM_PER_INCH * PRINT_DPI)

# Nature's final-submission guide: text other than panel letters is at
# most 7 pt and at least 5 pt. A frame caption is that text. 72 pt is
# one inch.
LABEL_POINT = 7
CAPTION_FLOOR_POINT = 5
POINTS_PER_INCH = 72

# The 12-6 potential 4*eps*[(sigma/r)**12 - (sigma/r)**6] is minimum at
# r = 2**(1/6) sigma. Coordinates for kind="cluster" are in units of sigma.
LJ_RMIN_OVER_SIGMA = 2.0 ** (1.0 / 6.0)
# Ideal fcc shells sit at r_min and r_min*sqrt(2). The cutoff is the midpoint.
LJ_SHELL_CUTOFF_OVER_SIGMA = LJ_RMIN_OVER_SIGMA * (1.0 + 2.0**0.5) / 2.0
# Half of r_min, so neighbours at the minimum touch.
LJ_CONTACT_RADIUS_OVER_SIGMA = LJ_RMIN_OVER_SIGMA / 2.0

# Kumar, Schmidt, and Skinner, J. Chem. Phys. 126, 204107 (2007):
# an O-H...O contact with r_OO at most 3.5 A and angle beta at most 30 deg.
HBOND_OO_MAX_A = 3.5
HBOND_BETA_MAX_DEG = 30.0

# Water is H2O. A hydrogen has atomic number 1. A geometric summit needs
# a donor oxygen and an acceptor oxygen.
HYDROGENS_PER_WATER = 2
HYDROGEN_Z = 1
DONOR_AND_ACCEPTOR = 2

# Jmol autobond, taken from OpenBabel (Hanson, JmolConstants): the contact
# must be at least the OpenBabel minimum and at most the sum of the covalent
# radii plus this tolerance. The radii are the Cordero set in ASE
# (Dalton Trans. 2008, 2832-2838).
BOND_TOLERANCE_A = 0.45
BOND_MIN_A = 0.4

# VTK's parallel scale is half the height of the orthographic window.
VTK_PARALLEL_SCALE = 0.5

CORAL = RUHI_COLORS["coral"]
TEAL = RUHI_COLORS["teal"]
SKY = RUHI_COLORS["sky"]
# Coral marks the highlight and sky the hydrogen-bond dash, which leaves
# three Ruhi colours for atom bodies. Hydrogen, carbon, and oxygen, the
# elements of water and malonaldehyde, take one each. The pairing is the
# one-to-one assignment with the smallest summed CIELAB distance to the
# Jmol colours. A fourth element would repeat a colour, so any other
# element needs a colour from the caller.
ELEMENT_COLORS: dict[str, str] = {
    "H": RUHI_COLORS["sunshine"],
    "C": TEAL,
    "O": RUHI_COLORS["magenta"],
}

# solvis turns both on by default. Depth peeling only orders translucent
# layers, and these spheres are opaque: on the LJ38 figure it changes no
# pixel. With shadows on, VTK 9.7.1 fails to compile the shader of each
# unlit outline, and the outline takes a dark tint of the atom colour in
# place of the ink. Shadows also darken about half of the cluster pixels
# and move the fill further from coral and teal.
SOLVIS_DEPTH_PEELING = False
SOLVIS_SHADOWS = False

# WCAG 2.1 SC 1.4.11 asks for 3:1 between a graphic and the colours next
# to it. On white, coral is 2.9:1 and sunshine 1.4:1, so every atom is
# outlined in the theme ink. paton's depth fog blends far outlines toward
# the ground. On the 64-water cell the faintest outline is #b2b2b2, 2.1:1
# on white, and 60 of 128 hydrogen-bond dashes fall under 3:1.
XYZRENDER_FOG = False
# Nature's final-submission guide sets line weights from 0.25 pt to 1 pt.
# The solvis outline takes the lower bound.
OUTLINE_POINT = 0.25

KIND_BACKEND = {
    "cluster": "solvis",
    "cell": "xyzrender",
    "instanton": "xyzrender",
    "neb": "xyzrender",
}

PATH_KINDS = frozenset({"instanton", "neb"})


def _hex_to_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


CORAL_RGB = _hex_to_rgb(CORAL)
TEAL_RGB = _hex_to_rgb(TEAL)


@dataclass
class Frame:
    """One oriented picture: positions are already in the camera frame."""

    symbols: list[str]
    positions: np.ndarray
    colors: np.ndarray
    radii: np.ndarray
    highlight: np.ndarray
    bonds: list[tuple[int, int]] = field(default_factory=list)
    hbonds: list[tuple[int, int]] = field(default_factory=list)
    label: str = ""
    explicit_bonds: bool = False


def covalent_radius(symbol: str) -> float:
    """Cordero covalent radius in angstroms."""
    from ase.data import atomic_numbers, covalent_radii

    return float(covalent_radii[atomic_numbers[symbol]])


def element_color(symbol: str, colors: Mapping[str, str] | None = None) -> str:
    """Hex body colour for one element from an explicit map.

    The default map is ``ELEMENT_COLORS``. A symbol outside the map raises.
    """
    table = ELEMENT_COLORS if colors is None else colors
    try:
        return table[symbol]
    except KeyError:
        msg = f"No body colour is set for {symbol}"
        raise KeyError(msg) from None


def element_rgb(symbol: str, colors: Mapping[str, str] | None = None) -> np.ndarray:
    """``element_color`` as an RGB triple in 0..1."""
    return _hex_to_rgb(element_color(symbol, colors))


def frame_color_overrides(
    symbols: list[str], colors: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Symbol-to-hex map for one frame. A symbol outside the map raises."""
    table = ELEMENT_COLORS if colors is None else colors
    missing = sorted({symbol for symbol in symbols if symbol not in table})
    if missing:
        msg = f"No body colour is set for {', '.join(missing)}"
        raise KeyError(msg)
    return {symbol: table[symbol] for symbol in dict.fromkeys(symbols)}


def pair_distance(positions: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(positions[i] - positions[j]))


def is_covalent_contact(
    symbols: list[str], positions: np.ndarray, i: int, j: int
) -> bool:
    """True when the pair falls inside the Jmol/OpenBabel autobond window."""
    dist = pair_distance(positions, i, j)
    limit = covalent_radius(symbols[i]) + covalent_radius(symbols[j]) + BOND_TOLERANCE_A
    return BOND_MIN_A <= dist <= limit


def covalent_bonds(symbols: list[str], positions: np.ndarray) -> list[tuple[int, int]]:
    """Undirected covalent pairs, i < j."""
    pairs = []
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            if is_covalent_contact(symbols, positions, i, j):
                pairs.append((i, j))
    return pairs


def hydrogen_bonds(symbols: list[str], positions: np.ndarray) -> list[tuple[int, int]]:
    """Kumar O-H...O contacts that are not themselves covalent bonds.

    There is no periodic boundary. Each pair is ``(hydrogen, acceptor)``.
    """
    covalent = {frozenset(pair) for pair in covalent_bonds(symbols, positions)}
    oxygens = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    hydrogens = [i for i, symbol in enumerate(symbols) if symbol == "H"]
    beta_max = np.deg2rad(HBOND_BETA_MAX_DEG)
    found: list[tuple[int, int]] = []
    for donor in oxygens:
        for hydrogen in hydrogens:
            if frozenset((donor, hydrogen)) not in covalent:
                continue
            oh = positions[hydrogen] - positions[donor]
            dist_oh = float(np.linalg.norm(oh))
            if dist_oh == 0.0:
                continue
            oh_hat = oh / dist_oh
            for acceptor in oxygens:
                if acceptor == donor or frozenset((hydrogen, acceptor)) in covalent:
                    continue
                oo = positions[acceptor] - positions[donor]
                dist = float(np.linalg.norm(oo))
                if dist > HBOND_OO_MAX_A or dist == 0.0:
                    continue
                beta = float(np.arccos(np.clip(np.dot(oh_hat, oo / dist), -1.0, 1.0)))
                if beta <= beta_max:
                    found.append((hydrogen, acceptor))
    return found


def kabsch(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation that maps ``mobile`` onto ``target``. Each array is (N, 3)."""
    h = (mobile - mobile.mean(0)).T @ (target - target.mean(0))
    _u, _, vt = np.linalg.svd(h)
    rot = vt.T @ _u.T
    if np.linalg.det(rot) < 0.0:
        vt = vt.copy()
        vt[-1] *= -1.0
        rot = vt.T @ _u.T
    return rot


def inertia_rotation(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Columns are the image x, y, and z axes.

    z is the largest principal moment, so a flat molecule is seen face on.
    """
    com = np.average(positions, axis=0, weights=masses)
    rel = positions - com
    inertia = np.zeros((3, 3))
    for mass, vec in zip(masses, rel, strict=True):
        inertia += mass * ((vec @ vec) * np.eye(3) - np.outer(vec, vec))
    values, vectors = np.linalg.eigh(inertia)
    order = np.argsort(values)
    rot = np.column_stack(
        [vectors[:, order[0]], vectors[:, order[1]], vectors[:, order[2]]]
    )
    if np.linalg.det(rot) < 0.0:
        rot[:, 0] *= -1.0
    return rot


def _masses(symbols: list[str]) -> np.ndarray:
    from ase.data import atomic_masses, atomic_numbers

    return np.array([atomic_masses[atomic_numbers[s]] for s in symbols], dtype=float)


def transferring_hydrogen(frames: list[tuple[list[str], np.ndarray]]) -> int:
    """Index of the hydrogen that moves farthest between the endpoints.

    The last frame is rotated onto the first by the heavy atoms (Z > 1)
    before the displacement is measured.
    """
    symbols = frames[0][0]
    from ase.data import atomic_numbers

    heavy = np.array([atomic_numbers[s] > HYDROGEN_Z for s in symbols])
    if not heavy.any():
        msg = "A path needs at least one atom heavier than hydrogen"
        raise ValueError(msg)
    target = frames[0][1][heavy]
    last = frames[-1][1]
    rot = kabsch(last[heavy], target)
    aligned = (last - last[heavy].mean(0)) @ rot + target.mean(0)
    delta = np.linalg.norm(aligned - frames[0][1], axis=1)
    delta[np.array([s != "H" for s in symbols])] = -1.0
    if float(delta.max()) < 0.0:
        msg = "A path needs a hydrogen"
        raise ValueError(msg)
    return int(np.argmax(delta))


def align_path(frames: list[tuple[list[str], np.ndarray]]) -> list[np.ndarray]:
    """Kabsch every frame onto the first, then one inertia rotation for all."""
    symbols = frames[0][0]
    from ase.data import atomic_numbers

    heavy = np.array([atomic_numbers[s] > HYDROGEN_Z for s in symbols])
    target = frames[0][1][heavy]
    aligned = [frames[0][1]]
    for _, pos in frames[1:]:
        rot = kabsch(pos[heavy], target)
        aligned.append((pos - pos[heavy].mean(0)) @ rot + target.mean(0))
    rot = inertia_rotation(aligned[0][heavy], _masses(symbols)[heavy])
    return [pos @ rot for pos in aligned]


def frame_indices(n: int, summit: int) -> list[int]:
    """Endpoints, the summit, and the index halfway along each side.

    Duplicates are dropped. The count is fixed by that construction.
    """

    def mid(a: int, b: int) -> int:
        return (a + b) // 2

    raw = [0, mid(0, summit), summit, mid(summit, n - 1), n - 1]
    out: list[int] = []
    for index in raw:
        if index not in out:
            out.append(index)
    return out


def geometric_summit(
    symbols: list[str], positions: list[np.ndarray], hydrogen: int
) -> int:
    """Frame where the transferring H is equidistant from donor and acceptor.

    The donor is the oxygen closest to that H in the first frame. The
    acceptor is the oxygen closest to it in the last frame.
    """
    oxygens = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    if len(oxygens) < DONOR_AND_ACCEPTOR:
        msg = "A geometric summit needs two oxygens"
        raise ValueError(msg)
    donor = min(
        oxygens, key=lambda i: np.linalg.norm(positions[0][i] - positions[0][hydrogen])
    )
    acceptor = min(
        oxygens, key=lambda i: np.linalg.norm(positions[-1][i] - positions[-1][hydrogen])
    )
    scores = []
    for pos in positions:
        d_donor = np.linalg.norm(pos[hydrogen] - pos[donor])
        d_acceptor = np.linalg.norm(pos[hydrogen] - pos[acceptor])
        scores.append(abs(d_donor - d_acceptor))
    return int(np.argmin(scores))


def pack_frames(frames: list[Frame]) -> float:
    """Centre every frame on the shared bounding box and return its longest side."""
    lo = np.array([np.inf, np.inf, np.inf])
    hi = -lo
    for frame in frames:
        lo = np.minimum(lo, (frame.positions - frame.radii[:, None]).min(0))
        hi = np.maximum(hi, (frame.positions + frame.radii[:, None]).max(0))
    centre = (lo + hi) / 2.0
    for frame in frames:
        frame.positions = frame.positions - centre
    return float(max(hi - lo))


def _symbols_positions(atoms) -> tuple[list[str], np.ndarray]:
    symbols = list(atoms.get_chemical_symbols())
    positions = np.array(atoms.positions, dtype=float, copy=True)
    return symbols, positions


def _as_structures(structure) -> list:
    if hasattr(structure, "get_chemical_symbols"):
        return [structure]
    structures = list(structure)
    if not structures:
        msg = "render_structure received an empty sequence"
        raise ValueError(msg)
    return structures


def backend_for(kind: str, structure) -> str:
    """Name of the renderer used for this kind and structure.

    ``snapshot`` follows the element count: one element uses the cluster
    backend, anything else uses the cell backend.
    """
    if kind == "snapshot":
        symbols = list(_as_structures(structure)[0].get_chemical_symbols())
        if len(set(symbols)) == 1:
            return KIND_BACKEND["cluster"]
        return KIND_BACKEND["cell"]
    if kind not in KIND_BACKEND:
        msg = f"Unknown structure kind {kind!r}"
        raise ValueError(msg)
    return KIND_BACKEND[kind]


def _paint(
    symbols: list[str],
    highlight: np.ndarray,
    body: np.ndarray | None,
    colors: Mapping[str, str] | None = None,
) -> np.ndarray:
    """RGB body colours. A missing element raises, as the override map does.

    ``body`` is one colour for every atom, used for a cluster. Otherwise
    each symbol has to be in the element map.
    """
    if body is not None:
        painted = np.broadcast_to(body, (len(symbols), 3)).copy()
    else:
        mapped = frame_color_overrides(symbols, colors)
        painted = np.vstack([_hex_to_rgb(mapped[symbol]) for symbol in symbols])
    painted[highlight] = CORAL_RGB
    return painted


def _lj_highlight(positions: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    coord = (dist < LJ_SHELL_CUTOFF_OVER_SIGMA).sum(axis=1)
    return coord == int(coord.min())


def _neighbor_highlight(symbols: list[str], positions: np.ndarray) -> np.ndarray:
    bonds = covalent_bonds(symbols, positions)
    coord = np.zeros(len(symbols), dtype=int)
    for i, j in bonds:
        coord[i] += 1
        coord[j] += 1
    return coord == int(coord.min())


def _user_highlight(
    n: int, highlight: list[int] | None, default: np.ndarray
) -> np.ndarray:
    if highlight is None:
        return default
    mask = np.zeros(n, dtype=bool)
    for index in highlight:
        if index < 0 or index >= n:
            msg = f"Highlight index {index} is outside 0..{n - 1}"
            raise ValueError(msg)
        mask[index] = True
    return mask


def _molecule_frame(
    symbols: list[str],
    positions: np.ndarray,
    hydrogen: int | None,
    label: str,
    highlight: list[int] | None,
    *,
    colors: Mapping[str, str] | None = None,
) -> Frame:
    bonds = covalent_bonds(symbols, positions)
    hbonds = hydrogen_bonds(symbols, positions)
    if highlight is None and hydrogen is not None:
        mask = np.zeros(len(symbols), dtype=bool)
        mask[hydrogen] = True
    else:
        mask = _user_highlight(
            len(symbols), highlight, np.zeros(len(symbols), dtype=bool)
        )
    painted = _paint(symbols, mask, None, colors)
    radii = np.array([covalent_radius(s) for s in symbols])
    return Frame(symbols, positions, painted, radii, mask, bonds, hbonds, label, False)


def _orthorhombic_edges(atoms) -> np.ndarray | None:
    cell = getattr(atoms, "cell", None)
    if cell is None or float(cell.volume) == 0.0 or not bool(cell.orthorhombic):
        return None
    return np.array(cell.lengths(), dtype=float)


def _water_molecules(
    symbols: list[str], positions: np.ndarray
) -> list[tuple[int, int, int]] | None:
    """Each oxygen and the two hydrogens covalently bound to it.

    Returns None when the stoichiometry is not water or an oxygen does not
    have two unused covalent hydrogens.
    """
    n_o = symbols.count("O")
    n_h = symbols.count("H")
    if n_o == 0 or n_h != HYDROGENS_PER_WATER * n_o or set(symbols) - {"O", "H"}:
        return None
    oxygens = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    hydrogens = [i for i, symbol in enumerate(symbols) if symbol == "H"]
    used: set[int] = set()
    molecules = []
    for oxygen in oxygens:
        bonded = [
            h
            for h in hydrogens
            if h not in used and is_covalent_contact(symbols, positions, oxygen, h)
        ]
        bonded.sort(key=lambda h: pair_distance(positions, oxygen, h))
        if len(bonded) < HYDROGENS_PER_WATER:
            return None
        pair = bonded[:HYDROGENS_PER_WATER]
        used.update(pair)
        molecules.append((oxygen, pair[0], pair[1]))
    return molecules


def _wrap_water(positions: np.ndarray, molecules, edges: np.ndarray) -> np.ndarray:
    out = positions.copy()
    for oxygen, h1, h2 in molecules:
        for hydrogen in (h1, h2):
            delta = out[hydrogen] - out[oxygen]
            delta -= edges * np.round(delta / edges)
            out[hydrogen] = out[oxygen] + delta
        shift = -edges * np.floor(out[oxygen] / edges)
        out[oxygen] += shift
        out[h1] += shift
        out[h2] += shift
    return out


def _periodic_hbonds(symbols, positions, molecules, edges):
    """Kumar contacts. An acceptor outside the cell is stored as a ghost oxygen."""
    positions = _wrap_water(positions, molecules, edges)
    bonds = []
    for oxygen, h1, h2 in molecules:
        bonds.append((oxygen, h1))
        bonds.append((oxygen, h2))
    oxygens = [oxygen for oxygen, _, _ in molecules]
    hbonds: list[tuple[int, int]] = []
    beta_max = np.deg2rad(HBOND_BETA_MAX_DEG)
    symbols = list(symbols)
    for oxygen, h1, h2 in molecules:
        donor = positions[oxygen]
        for hydrogen in (h1, h2):
            oh = positions[hydrogen] - donor
            oh_hat = oh / np.linalg.norm(oh)
            for acceptor in oxygens:
                if acceptor == oxygen:
                    continue
                oo = positions[acceptor] - donor
                oo -= edges * np.round(oo / edges)
                dist = float(np.linalg.norm(oo))
                if dist > HBOND_OO_MAX_A or dist == 0.0:
                    continue
                beta = float(np.arccos(np.clip(np.dot(oh_hat, oo / dist), -1.0, 1.0)))
                if beta > beta_max:
                    continue
                image = donor + oo
                # Closer than the shortest bond we draw, so this is the
                # acceptor itself and not a second atom.
                if np.linalg.norm(positions[acceptor] - image) < BOND_MIN_A:
                    hbonds.append((hydrogen, acceptor))
                else:
                    ghost = len(symbols)
                    symbols.append("O")
                    positions = np.vstack([positions, image])
                    hbonds.append((hydrogen, ghost))
    return symbols, positions, bonds, hbonds


def _front_water(
    molecules: list[tuple[int, int, int]],
    symbols: list[str],
    positions: np.ndarray,
    radii: np.ndarray,
    centre: np.ndarray,
) -> tuple[int, int, int]:
    """The water nearest ``centre`` in the image whose oxygen is in view.

    xyzrender draws a larger z nearer the camera. An oxygen is covered
    when a nearer oxygen, a ghost included, sits closer to it in the image
    than the sum of their radii. When every water is covered, every water
    is a candidate.
    """
    oxygens = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    xy = positions[oxygens, :2]
    gap = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    reach = radii[oxygens][:, None] + radii[oxygens][None, :]
    depth = positions[oxygens, 2]
    covered = ((gap < reach) & (depth[None, :] > depth[:, None])).any(axis=1)
    hidden = {oxygen for oxygen, flag in zip(oxygens, covered, strict=True) if flag}
    candidates = [mol for mol in molecules if mol[0] not in hidden] or molecules
    return min(
        candidates,
        key=lambda mol: float(np.linalg.norm(positions[mol[0], :2] - centre[:2])),
    )


def _water_frame(
    atoms,
    highlight: list[int] | None,
    label: str,
    colors: Mapping[str, str] | None = None,
) -> Frame | None:
    edges = _orthorhombic_edges(atoms)
    if edges is None:
        return None
    symbols, positions = _symbols_positions(atoms)
    molecules = _water_molecules(symbols, positions)
    if molecules is None:
        return None
    n_input = len(symbols)
    symbols, positions, bonds, hbonds = _periodic_hbonds(
        symbols, positions, molecules, edges
    )
    radii = np.array([covalent_radius(s) for s in symbols])
    if highlight is None:
        # The view is down c, so a water inside a column sits behind the
        # waters above it.
        chosen = _front_water(molecules, symbols, positions, radii, 0.5 * edges)
        mask = np.zeros(len(symbols), dtype=bool)
        mask[list(chosen)] = True
    else:
        # Indices address the atoms the caller passed. Ghosts are appended.
        mask = _user_highlight(n_input, highlight, np.zeros(n_input, dtype=bool))
        mask = np.pad(mask, (0, len(symbols) - n_input))
    painted = _paint(symbols, mask, None, colors)
    positions = positions - 0.5 * edges
    return Frame(symbols, positions, painted, radii, mask, bonds, hbonds, label, True)


def _cluster_frame(atoms, highlight: list[int] | None, label: str, *, lj: bool) -> Frame:
    symbols, positions = _symbols_positions(atoms)
    if lj:
        mask = _user_highlight(len(symbols), highlight, _lj_highlight(positions))
        colors = _paint(symbols, mask, TEAL_RGB)
        radii = np.full(len(symbols), LJ_CONTACT_RADIUS_OVER_SIGMA)
        masses = np.ones(len(symbols))
    else:
        mask = _user_highlight(
            len(symbols), highlight, _neighbor_highlight(symbols, positions)
        )
        colors = _paint(symbols, mask, TEAL_RGB)
        radii = np.array([covalent_radius(s) for s in symbols])
        masses = _masses(symbols)
    centred = positions - np.average(positions, axis=0, weights=masses)
    positions = centred @ inertia_rotation(centred, masses)
    return Frame(symbols, positions, colors, radii, mask, [], [], label, False)


def _path_frames(
    structures,
    kind: str,
    energies,
    highlight: list[int] | None,
    colors: Mapping[str, str] | None = None,
) -> list[Frame]:
    raw = [_symbols_positions(atoms) for atoms in structures]
    symbols = raw[0][0]
    if any(frame[0] != symbols for frame in raw):
        msg = "Every frame of a path must have the same atoms in the same order"
        raise ValueError(msg)
    hydrogen = transferring_hydrogen(raw)
    aligned = align_path(raw)
    if energies is None:
        summit = geometric_summit(symbols, aligned, hydrogen)
        summit_word = "summit" if kind == "instanton" else "saddle"
    else:
        values = np.array(list(energies), dtype=float)
        if len(values) != len(raw):
            msg = "energies must have one value per frame"
            raise ValueError(msg)
        summit = int(np.argmax(values))
        summit_word = "summit" if kind == "instanton" else "saddle"
    noun = "bead" if kind == "instanton" else "image"
    frames = []
    for index in frame_indices(len(raw), summit):
        label = f"{noun} {index}"
        if index == summit:
            label = f"{label} {summit_word}"
        frames.append(
            _molecule_frame(
                symbols, aligned[index], hydrogen, label, highlight, colors=colors
            )
        )
    return frames


def _single_frames(
    structures,
    kind: str,
    highlight: list[int] | None,
    colors: Mapping[str, str] | None = None,
) -> tuple[list[Frame], str]:
    """Return frames and the renderer name."""
    frames: list[Frame] = []
    renderer = backend_for(kind, structures[0])
    for number, atoms in enumerate(structures):
        label = str(atoms.info.get("label", "")) if hasattr(atoms, "info") else ""
        if not label:
            label = f"frame {number}" if len(structures) > 1 else ""
        if renderer == "solvis":
            frames.append(_cluster_frame(atoms, highlight, label, lj=(kind == "cluster")))
            continue
        water = (
            _water_frame(atoms, highlight, label, colors)
            if kind in {"cell", "snapshot"}
            else None
        )
        if water is not None:
            frames.append(water)
            continue
        symbols, positions = _symbols_positions(atoms)
        masses = _masses(symbols)
        centred = positions - np.average(positions, axis=0, weights=masses)
        positions = centred @ inertia_rotation(centred, masses)
        frames.append(
            _molecule_frame(symbols, positions, None, label, highlight, colors=colors)
        )
    return frames, renderer


def _import_xyzrender():
    """Import xyzrender via ensure_import, then a normal import."""
    try:
        from rgpycrumbs._aux import enable_library_auto_deps, ensure_import

        enable_library_auto_deps()
        return ensure_import("xyzrender")
    except ImportError:
        pass
    try:
        import xyzrender
    except ImportError as exc:
        msg = (
            "xyzrender is required for this figure. "
            "Install with: pip install 'xyzrender>=0.3.8'"
        )
        raise ImportError(msg) from exc
    return xyzrender


def _import_solvis():
    """Import solvis via ensure_import, then a normal import."""
    try:
        from rgpycrumbs._aux import enable_library_auto_deps, ensure_import

        enable_library_auto_deps()
        return ensure_import("solvis")
    except ImportError:
        pass
    try:
        import solvis
    except ImportError as exc:
        msg = (
            "solvis is required for a cluster. "
            "Install with: pip install 'chemparseplot[solvis]'"
        )
        raise ImportError(msg) from exc
    return solvis


def _caption_face(theme: PlotTheme) -> str:
    """Font file of the theme's caption face, else matplotlib's sans-serif."""
    from matplotlib.font_manager import FontProperties, findfont

    # A one-string family is parsed as a fontconfig pattern, which splits
    # "sans-serif" at the hyphen. A list is read as family names.
    try:
        return findfont(
            FontProperties(family=[theme.caption_font]), fallback_to_default=False
        )
    except ValueError:
        log.warning(
            "Caption font %r not found. Falling back to 'sans-serif'.",
            theme.caption_font,
        )
        return findfont(FontProperties(family=["sans-serif"]))


def _caption_px_range(dpi: float) -> tuple[int, int]:
    """Smallest and largest whole-pixel sizes that print from 5 pt to 7 pt.

    The smallest rounds up and the largest rounds down, so both stay
    inside the range.
    """
    low = math.ceil(CAPTION_FLOOR_POINT * dpi / POINTS_PER_INCH)
    high = math.floor(LABEL_POINT * dpi / POINTS_PER_INCH)
    if low > high:
        msg = f"No whole-pixel caption size prints from 5 pt to 7 pt at {dpi} dpi"
        raise ValueError(msg)
    return low, high


def _ellipsise(label: str, font, width: int) -> str:
    """Shorten ``label`` so it fits ``width``, ending in an ellipsis."""
    if font.getlength(label) <= width:
        return label
    mark = "…"
    if font.getlength(mark) > width:
        return ""
    lo = 0
    hi = len(label)
    best = mark
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = label[:mid].rstrip() + mark
        if font.getlength(candidate) <= width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _fit_caption(label: str, width: int, dpi: float, face: str) -> tuple[str, object]:
    """The largest size from 7 pt down to 5 pt at which ``label`` fits ``width``.

    A label that is wider than ``width`` at 5 pt is ellipsised at 5 pt.
    """
    from PIL import ImageFont

    low, high = _caption_px_range(dpi)
    for px in range(high, low, -1):
        font = ImageFont.truetype(face, px)
        if font.getlength(label) <= width:
            return label, font
    floor = ImageFont.truetype(face, low)
    return _ellipsise(label, floor, width), floor


def _theme_rgba(color: str) -> tuple[int, int, int, int]:
    from matplotlib.colors import to_rgba

    channels = to_rgba(color)
    return tuple(round(channel * 255) for channel in channels)


def _theme_hex(color: str) -> str:
    red, green, blue, _alpha = _theme_rgba(color)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _embedded_dpi(image) -> float | None:
    """Horizontal dpi stored on the image, or None when the file has none."""
    dpi = image.info.get("dpi")
    if not dpi:
        return None
    value = dpi[0] if isinstance(dpi, tuple) else dpi
    value = float(value)
    if value <= 0:
        return None
    return value


def _tag_print_dpi(image, layout_px: int):
    """Record a dpi when the raster file did not.

    The layout panel is ``layout_px`` at the print resolution. A writer
    that emits more pixels and no dpi chunk, which is what xyzrender
    0.3.8 does, is printing that same panel at a higher dpi.
    """
    if _embedded_dpi(image) is None:
        side = max(image.size)
        dpi = PRINT_DPI * side / float(layout_px)
        image.info["dpi"] = (dpi, dpi)
    return image


def _caption_px(image) -> int:
    """Pixel size of a 7 pt caption on this raster."""
    dpi = _embedded_dpi(image)
    if dpi is None:
        msg = "A caption needs the dpi of the panel"
        raise ValueError(msg)
    # Floor, so the printed size stays at most 7 pt. round(87.5) is 88.
    return _caption_px_range(dpi)[1]


def _stitch(images, labels: list[str], theme: PlotTheme) -> np.ndarray:
    from PIL import Image, ImageDraw

    if len(images) == 1 and not labels[0]:
        return np.asarray(images[0])
    dpi = _embedded_dpi(images[0])
    if dpi is None:
        msg = "A caption needs the dpi of the panel"
        raise ValueError(msg)
    face = _caption_face(theme)
    fitted = [
        _fit_caption(label, image.width, dpi, face)
        for image, label in zip(images, labels, strict=True)
    ]
    # getmetrics reports a positive descent. The row is the tallest em
    # box, so a descender such as the g in "image N" stays on the canvas.
    row = 0
    for _label, font in fitted:
        ascent, descent = font.getmetrics()
        row = max(row, ascent + abs(descent))
    width = sum(im.width for im in images)
    height = max(im.height for im in images) + row
    canvas = Image.new("RGBA", (width, height), _theme_rgba(theme.facecolor))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for image, (label, font) in zip(images, fitted, strict=True):
        canvas.paste(image, (x, 0))
        if label:
            ascent, _descent = font.getmetrics()
            draw.text(
                (x, image.height + ascent),
                label,
                fill=theme.textcolor,
                font=font,
                anchor="ls",
            )
        x += image.width
    return np.asarray(canvas)


def _hex(rgb: np.ndarray) -> str:
    vals = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(int)
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _render_solvis(
    frames: list[Frame], span: float, canvas: int, theme: PlotTheme
) -> np.ndarray:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    solvis = _import_solvis()
    import pyvista

    atomic_plotter = solvis.visualization.AtomicPlotter
    outline_px = OUTLINE_POINT * PRINT_DPI / POINTS_PER_INCH
    if canvas <= 2.0 * outline_px:
        msg = "A cluster panel must be wider than two outlines"
        raise ValueError(msg)
    # The view grows by one outline on each side, so an atom on the edge
    # of the bounding box keeps its whole outline on the canvas.
    view = span * canvas / (canvas - 2.0 * outline_px)
    outline = outline_px * view / canvas
    ink = _theme_hex(theme.textcolor)

    images = []
    for frame in frames:
        plotter = atomic_plotter(
            interactive_mode=False,
            window_size=[canvas, canvas],
            depth_peeling=SOLVIS_DEPTH_PEELING,
            shadows=SOLVIS_SHADOWS,
        )
        plotter.plotter.set_background(theme.facecolor)
        for index, (pos, color, radius) in enumerate(
            zip(frame.positions, frame.colors, frame.radii, strict=True)
        ):
            plotter.add_single_atom_as_sphere(
                pos, _hex(color), actor_name=f"a{index}", radius=float(radius)
            )
            # A sphere one outline wider, drawn from its back faces only,
            # shows as a ring around the atom.
            plotter.plotter.add_mesh(
                pyvista.Sphere(radius=float(radius) + outline, center=pos),
                color=ink,
                culling="front",
                lighting=False,
            )
        camera = plotter.plotter.camera
        camera.parallel_projection = True
        camera.focal_point = (0.0, 0.0, 0.0)
        camera.position = (0.0, 0.0, span)
        camera.up = (0.0, 1.0, 0.0)
        camera.parallel_scale = view * VTK_PARALLEL_SCALE
        image = plotter.plotter.screenshot(return_img=True)
        plotter.close()
        rgba = np.concatenate(
            [image, np.full((*image.shape[:2], 1), 255, np.uint8)], axis=-1
        )
        from PIL import Image

        images.append(_tag_print_dpi(Image.fromarray(rgba), canvas))
    return _stitch(images, [frame.label for frame in frames], theme)


def _render_xyzrender(
    frames: list[Frame],
    span: float,
    canvas: int,
    colors: Mapping[str, str] | None = None,
    theme: PlotTheme | None = None,
) -> np.ndarray:
    if theme is None:
        theme = RUHI_THEME
    xyzrender = _import_xyzrender()
    from PIL import Image
    from xyzrender.config import build_config

    images = []
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        for number, frame in enumerate(frames):
            xyz = root / f"{number}.xyz"
            png = root / f"{number}.png"
            lines = [str(len(frame.symbols)), frame.label or "structure"]
            for symbol, pos in zip(frame.symbols, frame.positions, strict=True):
                lines.append(f"{symbol:2s} {pos[0]: .8f} {pos[1]: .8f} {pos[2]: .8f}")
            xyz.write_text("\n".join(lines) + "\n")
            cfg = build_config(
                "paton",
                canvas_size=canvas,
                orient=False,
                background=_theme_hex(theme.facecolor),
                fog=XYZRENDER_FOG,
                hy=True,
                nci_color=SKY,
            )
            # paton is the line style. Body colours come from the element
            # map, and the outline is the theme ink.
            cfg.color_overrides = frame_color_overrides(frame.symbols, colors)
            cfg.atom_stroke_color = _theme_hex(theme.textcolor)
            cfg.fixed_span = span
            cfg.fixed_center = (0.0, 0.0)
            extra: dict = {}
            if frame.explicit_bonds:
                extra["unbond"] = ["all"]
                extra["bond"] = [f"{i + 1}-{j + 1}" for i, j in frame.bonds]
            nci = [(i + 1, j + 1) for i, j in frame.hbonds]
            ids = [str(i + 1) for i in np.flatnonzero(frame.highlight)]
            xyzrender.render(
                xyz,
                config=cfg,
                nci_bonds=nci or None,
                highlight=[(",".join(ids), CORAL)] if ids else None,
                output=png,
                transparent=False,
                orient=False,
                hy=True,
                **extra,
            )
            images.append(_tag_print_dpi(Image.open(png).convert("RGBA"), canvas))
    return _stitch(images, [frame.label for frame in frames], theme)


def render_structure(
    structure,
    *,
    kind: str = "snapshot",
    energies=None,
    highlight: list[int] | None = None,
    output: str | Path | None = None,
    canvas: int | None = None,
    element_colors: Mapping[str, str] | None = None,
    theme: PlotTheme | None = None,
) -> np.ndarray:
    """Draw a structure, or a strip of a path, and return an RGBA image.

    Parameters
    ----------
    structure:
        An ASE ``Atoms`` object, or a sequence of them in path order.
    kind:
        ``cluster`` (Lennard-Jones, coordinates in units of sigma),
        ``cell``, ``instanton``, ``neb``, or ``snapshot``. A snapshot
        with one element is drawn as a cluster of Cordero spheres. Any
        other snapshot is drawn as a molecule or a cell.
    energies:
        One energy per frame. The highest frame is the summit. Without
        energies, an instanton or NEB summit is the frame where the
        transferring hydrogen is equidistant from its donor and acceptor.
    highlight:
        Zero-based atom indices painted in Ruhi coral. The default for a
        cluster is the lowest-coordination shell, for a water cell the
        molecule nearest the cell centre in the image whose oxygen no
        nearer oxygen covers, and for a path the transferring hydrogen.
    output:
        Optional PNG path.
    canvas:
        Figure width in pixels. The default is a Nature single column
        (89 mm) at 300 dpi. Each panel is that width divided by the
        panel count, so a strip printed at one column keeps the caption
        inside Nature's 5 pt to 7 pt range. A caption is set at the
        largest whole-pixel size in that range at which it fits its
        panel. A label wider than its panel at 5 pt is ellipsised.
        xyzrender writes a PNG with more pixels than the panel and no
        dpi chunk. The caption uses the dpi of that file, or the print
        resolution times the measured pixel scale when the chunk is
        absent.
    element_colors:
        Map from element symbol to hex colour. The default is
        ``ELEMENT_COLORS``, three distinct Ruhi colours for hydrogen,
        carbon, and oxygen. A map must name every element in the
        figure, so a figure with any other element needs this map.
    theme:
        Caption face, ground, and ink. The default is the ruhi theme.
        Every atom is outlined in the ink. A ``caption_font`` that is
        not installed logs a warning naming the face, and the caption
        falls back to matplotlib's sans-serif face.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(height, width, 4)``.
    """
    structures = _as_structures(structure)
    if kind in PATH_KINDS:
        frames = _path_frames(structures, kind, energies, highlight, element_colors)
        renderer = KIND_BACKEND[kind]
    elif kind in KIND_BACKEND or kind == "snapshot":
        frames, renderer = _single_frames(structures, kind, highlight, element_colors)
    else:
        msg = f"Unknown structure kind {kind!r}"
        raise ValueError(msg)
    theme = RUHI_THEME if theme is None else theme
    span = pack_frames(frames)
    figure_px = CANVAS if canvas is None else int(canvas)
    # One printed figure. A strip divides that width across its panels.
    side = figure_px // len(frames)
    if side < 1:
        msg = "The figure width does not cover every panel"
        raise ValueError(msg)
    if renderer == "solvis":
        image = _render_solvis(frames, span, side, theme)
    elif renderer == "xyzrender":
        image = _render_xyzrender(frames, span, side, element_colors, theme)
    else:
        msg = f"Unknown renderer {renderer!r}"
        raise ValueError(msg)
    if output is not None:
        from PIL import Image

        Image.fromarray(image).save(output)
    return image
