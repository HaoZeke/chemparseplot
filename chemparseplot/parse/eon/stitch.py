# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Stitch multiple NEB segments into one continuous reaction band.

Segments share an absolute-energy axis (for example BLYP/3-21G) but carry small
minimization offsets at the structures they share. Concatenating them into one
band requires three steps:

1. **Slice** each segment to the frames of interest.
2. **Deduplicate** the junction frame: the first frame of every later segment
   repeats the last frame of the previous segment, so it is dropped.
3. **Align** each later segment by a constant energy shift so the (dropped)
   junction frame matches the previous segment's last kept frame, giving an
   unbroken energy profile.

Everything is referenced to the first frame of the first segment (reactant = 0).

``stitch_neb_segments`` writes a combined band (``neb.con``, ``neb_path_000.con``,
``neb_000.dat``, ``sp.con``) that the existing profile/landscape plotters consume
unchanged, and returns a :class:`StitchSummary` describing the boundaries and
per-segment barriers.

```{versionadded} 1.8.0
```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StitchedSegment:
    """One stitched segment positioned in the combined band.

    ``start`` / ``end`` are the combined-band indices (inclusive) of the frames
    this segment contributes after junction deduplication. ``well_energy`` is the
    segment's entry minimum and ``barrier`` is ``peak_energy - well_energy``, all
    in eV relative to the global reactant.
    """

    label: str
    start: int
    end: int
    well_energy: float
    peak_energy: float
    barrier: float


@dataclass(frozen=True, slots=True)
class StitchSummary:
    """Summary of a stitched multi-segment NEB band."""

    out_dir: Path
    n_frames: int
    boundary_indices: list[int]
    segments: list[StitchedSegment]
    highest_energy: float
    highest_index: int


def _cartesian_rmsd(atoms_a, atoms_b) -> float:
    """Mass-agnostic Cartesian RMSD between two frames with matching ordering."""
    pos_a = atoms_a.get_positions()
    pos_b = atoms_b.get_positions()
    return float(np.sqrt(np.mean(np.sum((pos_a - pos_b) ** 2, axis=1))))


def _cumulative_rmsd(atoms_list) -> list[float]:
    """Cumulative consecutive-frame RMSD, starting at 0 for the first frame."""
    coords = [0.0]
    for prev, cur in zip(atoms_list[:-1], atoms_list[1:], strict=True):
        coords.append(coords[-1] + _cartesian_rmsd(prev, cur))
    return coords


def stitch_neb_segments(
    segments: list[tuple[str, str | Path, int | None, int | None]],
    out_dir: str | Path,
    saddle_overrides: dict[str, tuple[str | Path, float]] | None = None,
) -> StitchSummary:
    """Stitch ordered NEB segments into one continuous band on disk.

    ```{versionadded} 1.8.0
    ```

    Parameters
    ----------
    segments
        Ordered ``(label, con_path, start, end)`` slices. ``start``/``end`` are
        Python slice bounds into the segment's frames (``frames[start:end]``);
        ``None`` means the natural end. The first frame of every segment after
        the first is treated as a duplicate junction and dropped after the
        alignment shift is computed from it.
    out_dir
        Directory to write ``neb.con``, ``neb_path_000.con``, ``neb_000.dat`` and
        ``sp.con`` into. Created if missing.
    saddle_overrides
        Optional ``{label: (saddle_con_path, saddle_energy_abs)}``. For the named
        segment the reported barrier uses ``saddle_energy_abs`` (in the segments'
        absolute energy scale, e.g. from a dimer refinement) instead of the band
        maximum, and the saddle geometry is written to ``sp.con``.

    Returns
    -------
    StitchSummary
        Boundary indices, per-segment barriers, and the overall highest point.
    """
    from readcon import read_con, write_con

    saddle_overrides = saddle_overrides or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_frames = []  # ConFrame objects forming the combined band
    seg_records: list[StitchedSegment] = []
    boundary_indices: list[int] = []

    ref: float | None = None  # global reference energy (first frame, absolute)
    prev_last_aligned: float | None = None  # previous segment last kept energy (abs)
    sp_best: tuple[float, object] | None = None  # (referenced energy, saddle atoms)

    for seg_idx, (label, con_path, start, end) in enumerate(segments):
        frames = read_con(str(con_path))
        seg_frames = list(frames[start:end])
        if not seg_frames:
            msg = f"Segment '{label}' ({con_path}) sliced to zero frames."
            raise ValueError(msg)

        seg_abs = np.array([f.energy for f in seg_frames], dtype=float)

        if seg_idx == 0:
            shift = 0.0
            ref = float(seg_abs[0])
        else:
            # Align the duplicate junction frame to the previous last kept frame.
            shift = float(prev_last_aligned) - float(seg_abs[0])

        aligned = seg_abs + shift
        seg_ref = aligned - ref  # energies vs the global reactant

        prev_last_aligned = float(aligned[-1])

        # Drop the duplicate junction frame for every segment after the first.
        keep_from = 0 if seg_idx == 0 else 1
        kept_frames = seg_frames[keep_from:]
        kept_ref = seg_ref[keep_from:]

        seg_start = len(out_frames)
        for frame, energy in zip(kept_frames, kept_ref, strict=True):
            frame.set_energy(float(energy))
            out_frames.append(frame)
        seg_end = len(out_frames) - 1
        boundary_indices.append(seg_start)

        # Barrier: peak vs the segment's entry minimum. The entry minimum is the
        # segment's first frame (the junction), even when it is deduplicated.
        well_energy = float(seg_ref[0])
        if label in saddle_overrides:
            sad_path, sad_abs = saddle_overrides[label]
            peak_energy = float(sad_abs) + shift - ref
            sad_atoms = read_con(str(sad_path))[0].to_ase()
            if sp_best is None or peak_energy > sp_best[0]:
                sp_best = (peak_energy, sad_atoms)
        else:
            peak_energy = float(seg_ref.max())

        seg_records.append(
            StitchedSegment(
                label=label,
                start=seg_start,
                end=seg_end,
                well_energy=well_energy,
                peak_energy=peak_energy,
                barrier=peak_energy - well_energy,
            )
        )

    # --- Write the combined band ---
    write_con(str(out_dir / "neb.con"), out_frames)
    write_con(str(out_dir / "neb_path_000.con"), out_frames)

    atoms_list = [f.to_ase() for f in out_frames]
    rxn_coord = _cumulative_rmsd(atoms_list)
    energies = [float(f.energy) for f in out_frames]

    dat_path = out_dir / "neb_000.dat"
    with dat_path.open("w") as fh:
        fh.write(f"{'img':>4} {'rxn_coord':>12} {'energy':>12} {'f_para':>12}\n")
        for img, (rc, en) in enumerate(zip(rxn_coord, energies, strict=True)):
            fh.write(f"{img:>4d} {rc:>12.6f} {en:>12.6f} {0.0:>12.6f}\n")

    # --- Saddle overlay (sp.con) ---
    if sp_best is not None:
        from readcon import ConFrame

        sp_frame = ConFrame.from_ase(sp_best[1])
        sp_frame.set_energy(float(sp_best[0]))
        write_con(str(out_dir / "sp.con"), [sp_frame])
    else:
        # Fallback: the global band maximum so the landscape overlay still works.
        peak_idx = int(np.argmax(energies))
        write_con(str(out_dir / "sp.con"), [out_frames[peak_idx]])

    highest_index = int(np.argmax(energies))
    summary = StitchSummary(
        out_dir=out_dir,
        n_frames=len(out_frames),
        boundary_indices=boundary_indices,
        segments=seg_records,
        highest_energy=float(energies[highest_index]),
        highest_index=highest_index,
    )

    log.info(
        "Stitched %d segments -> %d frames; highest %.4f eV at image %d",
        len(seg_records),
        summary.n_frames,
        summary.highest_energy,
        summary.highest_index,
    )
    for rec in seg_records:
        log.info(
            "  %-16s frames %d..%d  barrier %.4f eV",
            rec.label,
            rec.start,
            rec.end,
            rec.barrier,
        )
    return summary
