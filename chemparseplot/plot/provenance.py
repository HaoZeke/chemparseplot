# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Deterministic figure files that record where they came from.

:func:`save_with_provenance` writes PNG, PDF or SVG with fonts embedded, no
timestamps and a fixed SVG id salt, so equal inputs give equal bytes. The file
carries a JSON description with the SHA-256 of each input and the versions of
the tools that drew it.

.. versionadded:: 1.10.0
"""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

import matplotlib as mpl

_TOOLS = ("chemparseplot", "rgpycrumbs", "matplotlib", "numpy")
_SVG_SALT = "chemparseplot"
_CREATOR = "chemparseplot"


def tool_versions(extra: tuple[str, ...] = ()) -> dict[str, str]:
    """Installed versions of the plotting stack; absent packages are omitted."""
    out = {}
    for name in (*_TOOLS, *extra):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(
    inputs: dict[str, str | Path],
    command: str = "",
    hashes: dict[str, str] | None = None,
) -> dict:
    """Provenance record: input hashes (by name), command line, tool versions.

    ``inputs`` are files to hash; ``hashes`` are digests already computed
    (a parser's ``provenance``), merged under the same key space.
    """
    return {
        "inputs": {
            **{k: file_sha256(v) for k, v in inputs.items()},
            **(hashes or {}),
        },
        "command": command,
        "tools": tool_versions(),
    }


def save_with_provenance(
    fig,
    output: str | Path,
    inputs: dict[str, str | Path],
    *,
    command: str = "",
    hashes: dict[str, str] | None = None,
    dpi: int = 200,
    sidecar: bool = True,
) -> dict:
    """Save ``fig`` deterministically and return the provenance record.

    The format follows the suffix (``.png``, ``.pdf``, ``.svg``). With
    ``hashes`` adds digests computed elsewhere. With ``sidecar`` a ``<output>.provenance.json`` is written beside the figure.
    """
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = describe(inputs, command, hashes)
    text = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    fmt = out.suffix.lstrip(".").lower()
    meta: dict
    if fmt == "png":
        meta = {"Software": _CREATOR, "Description": text}
    elif fmt == "pdf":
        meta = {
            "Creator": _CREATOR,
            "Producer": _CREATOR,
            "Subject": text,
            "CreationDate": None,
        }
    elif fmt == "svg":
        meta = {"Creator": _CREATOR, "Date": None, "Description": text}
    else:
        msg = f"unsupported figure format {fmt!r}"
        raise ValueError(msg)
    with mpl.rc_context(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "svg.hashsalt": _SVG_SALT,
        }
    ):
        fig.savefig(out, dpi=dpi, format=fmt, metadata=meta)
    rec["output"] = {"file": out.name, "sha256": file_sha256(out)}
    if sidecar:
        out.with_name(out.name + ".provenance.json").write_text(
            json.dumps(rec, indent=1, sort_keys=True) + "\n"
        )
    return rec


__all__ = ["describe", "file_sha256", "save_with_provenance", "tool_versions"]
