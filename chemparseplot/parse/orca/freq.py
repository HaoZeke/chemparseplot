# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""ORCA vibrational frequencies and IR spectrum parsing.

Text path ports the wailord ``orcaVis.vib_freq`` / ``ir_spec`` extractors.
When ``orca-pi`` (import name ``opi``) is installed and a job basename is
available, IR modes can also be read via :meth:`opi.output.core.Output.get_ir`
(official OPI IR grepping / ``IrMode``).

.. versionadded:: 1.9.11
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chemparseplot.parse.orca._opi import opi_available

__all__ = [
    "IrSpectrumMode",
    "VibMode",
    "parse_orca_ir_spectrum",
    "parse_orca_vibrational_frequencies",
]

_Backend = Literal["auto", "opi", "text"]

_VIB_HDR = re.compile(r"VIBRATIONAL FREQUENCIES")
_IR_HDR = re.compile(r"IR SPECTRUM")
_VIB_LINE = re.compile(r"^\s*(\d+):\s+(-?\d+\.?\d*)\s+cm\*\*-1(.*)$")


@dataclass(frozen=True, slots=True)
class VibMode:
    """One harmonic vibrational frequency line from ORCA."""

    mode: int
    freq_cm1: float
    imaginary: bool


@dataclass(frozen=True, slots=True)
class IrSpectrumMode:
    """One IR spectrum row (freq + intensity + dipole derivatives)."""

    mode: int
    freq_cm1: float
    intensity_km_mol: float
    dipole: tuple[float, float, float]
    eps: float | None = None


# A short string with no newline is probed as a path. The gaps are how far
# a table header may sit above its first data row. An ORCA IR row is mode,
# frequency, epsilon, intensity, then the dipole. A shorter row is mode,
# frequency, intensity, and three dipole components.
_PATH_PROBE_MAX = 4096
_VIB_HEADER_GAP = 20
_IR_HEADER_GAP = 30
_IR_ORCA_SCALARS = 4
_IR_FALLBACK_FIELDS = 6


def _as_text(path_or_text: str | Path) -> str:
    if isinstance(path_or_text, Path):
        return path_or_text.read_text(encoding="utf-8", errors="replace")
    s = str(path_or_text)
    if "\n" not in s and len(s) < _PATH_PROBE_MAX:
        p = Path(s)
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return s


def parse_orca_vibrational_frequencies(
    path_or_text: str | Path,
) -> list[VibMode]:
    """Parse the ``VIBRATIONAL FREQUENCIES`` block from ORCA text output."""
    text = _as_text(path_or_text)
    lines = text.splitlines()
    modes: list[VibMode] = []
    for i, line in enumerate(lines):
        if not _VIB_HDR.search(line):
            continue
        j = i + 1
        while j < len(lines) and not _VIB_LINE.match(lines[j]):
            j += 1
            if j - i > _VIB_HEADER_GAP:
                break
        while j < len(lines):
            m = _VIB_LINE.match(lines[j])
            if not m:
                break
            freq = float(m.group(2))
            rest = m.group(3) or ""
            imag = freq < 0 or "imaginary" in rest.lower()
            modes.append(VibMode(mode=int(m.group(1)), freq_cm1=freq, imaginary=imag))
            j += 1
        if modes:
            break
    return modes


def _ir_mode_from_orca_line(raw_line: str) -> IrSpectrumMode:
    """Parse one ORCA IR row: mode, frequency, epsilon, intensity, dipole.

    This is the column order of ``opi.output.ir_mode.IrMode.from_string``.
    The text parser does not import opi, so a machine without that package
    still reads the table.
    """
    left, right = raw_line.split("(", maxsplit=1)
    tx, ty, tz = (float(part) for part in right.split(")", maxsplit=1)[0].split())
    parts = left.replace(":", "").split()
    if len(parts) < _IR_ORCA_SCALARS:
        msg = "IR row has no epsilon and intensity before the dipole"
        raise ValueError(msg)
    return IrSpectrumMode(
        mode=int(parts[0]),
        freq_cm1=float(parts[1]),
        intensity_km_mol=float(parts[3]),
        dipole=(tx, ty, tz),
        eps=float(parts[2]),
    )


def _ir_mode_from_tokens(raw_line: str) -> IrSpectrumMode:
    """Parse a shorter row: mode, frequency, intensity, then the dipole."""
    cleaned = raw_line.replace(":", " ").replace("(", " ").replace(")", " ").split()
    if len(cleaned) < _IR_FALLBACK_FIELDS:
        msg = "IR row is shorter than mode, frequency, intensity, and a dipole"
        raise ValueError(msg)
    return IrSpectrumMode(
        mode=int(cleaned[0]),
        freq_cm1=float(cleaned[1]),
        intensity_km_mol=float(cleaned[2]),
        dipole=(float(cleaned[3]), float(cleaned[4]), float(cleaned[5])),
        eps=None,
    )


def _parse_ir_text(text: str) -> list[IrSpectrumMode]:
    lines = text.splitlines()
    out: list[IrSpectrumMode] = []
    for i, line in enumerate(lines):
        if not _IR_HDR.search(line):
            continue
        j = i + 1
        while (
            j < len(lines)
            and "Mode" not in lines[j]
            and not re.match(r"^\s*\d+:", lines[j])
        ):
            j += 1
            if j - i > _IR_HEADER_GAP:
                break
        if j < len(lines) and "Mode" in lines[j]:
            j += 1
        while j < len(lines) and set(lines[j].strip()) <= {"-", "="}:
            j += 1
        while j < len(lines):
            raw_line = lines[j]
            if raw_line.strip() == "" or set(raw_line.strip()) <= {"-"}:
                break
            if not re.match(r"^\s*\d+", raw_line):
                break
            try:
                out.append(_ir_mode_from_orca_line(raw_line))
            except ValueError:
                try:
                    out.append(_ir_mode_from_tokens(raw_line))
                except ValueError:
                    break
            j += 1
        if out:
            break
    return out


def _parse_ir_opi(basename: str, working_dir: Path) -> list[IrSpectrumMode]:
    from chemparseplot.parse.orca._opi import get_opi_output_class

    Output = get_opi_output_class()
    out = Output(basename, working_dir=working_dir)
    ir_dict = out.get_ir()
    if not ir_dict:
        return []
    modes: list[IrSpectrumMode] = []
    for mode_num, im in sorted(ir_dict.items()):
        modes.append(
            IrSpectrumMode(
                mode=int(mode_num),
                freq_cm1=float(im.wavenumber),
                intensity_km_mol=float(im.intensity),
                dipole=tuple(float(x) for x in im.dipole),
                eps=float(im.eps) if getattr(im, "eps", None) is not None else None,
            )
        )
    return modes


def parse_orca_ir_spectrum(
    path_or_text: str | Path,
    *,
    backend: _Backend = "auto",
    basename: str | None = None,
    working_dir: str | Path | None = None,
) -> list[IrSpectrumMode]:
    """Parse IR SPECTRUM (text) or OPI ``Output.get_ir`` when configured."""
    if backend in ("auto", "opi") and basename is not None:
        if opi_available():
            wd = Path(working_dir) if working_dir is not None else Path.cwd()
            modes = _parse_ir_opi(basename, wd)
            if modes or backend == "opi":
                return modes
        elif backend == "opi":
            from chemparseplot.parse.orca._opi import get_opi_output_class

            get_opi_output_class()
    if backend == "opi":
        return []
    return _parse_ir_text(_as_text(path_or_text))
