# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Parsers for the artifacts of gpr_optim campaign records.

A record is ``<record>/<set>/<cell>/`` holding ``result.json``,
``band/{band.h5,gprn.log,obs-archive/kernel.txt,mlflow.run.json}`` and, for
dimer cells, ``saddle/mlflow.run.json``. The MLflow file store sits at
``<record>/mlruns``. ``h5py`` is imported on use.

.. versionadded:: 1.10.0
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from chemparseplot.parse.surrogate.model import (
    AcquisitionEvent,
    BandHistory,
    BandSnapshot,
    CampaignTable,
    CellRecord,
    EvaluatedPoints,
    SearchHistory,
    SingleEndedHistory,
    SurrogateSearch,
)

_TAG = re.compile(r"^\[([A-Za-z0-9_-]+)\]\s+(.*)$")
_LEDGER = re.compile(r"Oracle calls \(to convergence\):\s*(\d+)\s*\(([^)]*)\)")
# Distance (A, Cartesian) under which a retained observation sits on an image.
MATCH_TOLERANCE = 0.02
# Window (ms) pairing a metric to the dimer step logged beside it.
PAIR_WINDOW_MS = 5000
# Calls added in one outer step above which the step is a spectrum batch.
SPECTRUM_BATCH_CALLS = 20
_MIN_TOKENS = 2


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _value(text: str):
    # Output of concurrent ranks can interleave; cut at the next tag.
    text = text.split("[", 1)[0]
    try:
        return float(text)
    except ValueError:
        return text


def parse_gprn_log(path: str | Path) -> dict[str, list[dict]]:
    """Group the ``[tag] key=value ...`` lines of a ``gprn``/``gprd`` log by tag.

    Bare words after the keys (``rejected``) are kept under ``flags``.
    Lines of other shapes are skipped.
    """
    out: dict[str, list[dict]] = {}
    with Path(path).open(errors="replace") as fh:
        for line in fh:
            if not line.startswith("["):
                continue
            m = _TAG.match(line.rstrip("\n"))
            if not m:
                continue
            fields: dict = {}
            flags: list[str] = []
            for tok in m.group(2).split():
                if tok.startswith("["):
                    break
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    fields[k] = _value(v)
                else:
                    flags.append(tok)
            if flags:
                fields["flags"] = flags
            out.setdefault(m.group(1), []).append(fields)
    return out


def read_mlflow_metrics(run_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Read an MLflow file-store run: ``name -> {time_ms, value, step}``."""
    metrics: dict[str, dict[str, np.ndarray]] = {}
    for f in sorted((Path(run_dir) / "metrics").glob("*")):
        rows = [ln.split() for ln in f.read_text().splitlines() if ln.strip()]
        if not rows:
            continue
        a = np.array([[float(x) for x in r[:3]] for r in rows])
        metrics[f.name] = {"time_ms": a[:, 0], "value": a[:, 1], "step": a[:, 2]}
    return metrics


def _locate_run(cell: Path, sub: str, mlruns: Path | None) -> Path | None:
    meta = cell / sub / "mlflow.run.json"
    if not meta.is_file():
        return None
    info = json.loads(meta.read_text())
    root = mlruns or cell.parents[1] / "mlruns"
    run = root / str(info["experiment_id"]) / str(info["run_id"])
    return run if run.is_dir() else None


def _tail(path: Path, nbytes: int = 1 << 17) -> str:
    with path.open("rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - nbytes))
        return fh.read().decode(errors="replace")


def _ledger(*logs: Path) -> dict[str, int]:
    ledger: dict[str, int] = {}
    for log in logs:
        if not log.is_file():
            continue
        found = list(_LEDGER.finditer(_tail(log)))
        m = found[-1] if found else None
        if m:
            ledger["total"] = int(m.group(1))
            for part in m.group(2).split(","):
                k, _, v = part.partition("=")
                if v.strip().isdigit() and int(v) > 0:
                    ledger[k.strip()] = ledger.get(k.strip(), 0) + int(v)
    return ledger


def _add_dimer_calls(rec: CellRecord, cell: Path) -> None:
    """Add the dimer stage to a band ledger so that it sums to the search calls."""
    res = json.loads((cell / "result.json").read_text())
    calls = (res.get("dimer") or {}).get("calls")
    if calls:
        rec.ledger["dimer"] = int(calls)
        rec.ledger["total"] = int(res.get("search_calls", rec.ledger.get("total", 0)))


def parse_gpr_optim_result(result_json: str | Path) -> CellRecord:
    """Build a :class:`CellRecord` from a cell's ``result.json``."""
    r = json.loads(Path(result_json).read_text())
    case = r.get("case", {})
    val = r.get("validation", {})
    wall = {}
    for key in ("band", "dimer"):
        if isinstance(r.get(key), dict) and "wall_s" in r[key]:
            wall[key] = float(r[key]["wall_s"])
    wall["validation"] = sum(
        float(r[k]["validation_wall_s"])
        for k in ("band", "dimer")
        if isinstance(r.get(k), dict) and "validation_wall_s" in r[k]
    )
    if "pipeline_wall_s" in r:
        wall["pipeline"] = float(r["pipeline_wall_s"])
    index = val.get("n_imaginary_above_frequency_tolerance", val.get("n_imaginary"))
    return CellRecord(
        label=case.get("label", Path(result_json).parent.name),
        set=case.get("set", ""),
        method=r.get("method", ""),
        status=r.get("status", ""),
        converged=r.get("converged"),
        passed=r.get("passed"),
        index=None if index is None else int(index),
        search_calls=r.get("search_calls"),
        validation_calls=r.get("validation_calls"),
        total_calls=r.get("total_oracle_calls"),
        wall=wall,
        energy_delta=(r.get("identity") or {}).get("delta_eV"),
        extras={
            "force_tolerance": r.get("settings", {}).get("force_tolerance"),
            "source_revision": r.get("runtime", {}).get("source_revision"),
            "acceptance_rule": r.get("acceptance_rule"),
        },
    )


def parse_gpr_optim_campaign(
    record_dir: str | Path, name: str | None = None
) -> CampaignTable:
    """Collect every ``<set>/<cell>/result.json`` of a record."""
    rec = Path(record_dir)
    cells = []
    for res in sorted(rec.glob("*/*/result.json")):
        c = parse_gpr_optim_result(res)
        c.ledger = _ledger(res.parent / "band" / "gprn.log")
        _add_dimer_calls(c, res.parent)
        cells.append(c)
    tol = next(
        (c.extras["force_tolerance"] for c in cells if c.extras.get("force_tolerance")),
        None,
    )
    return CampaignTable(name=name or rec.name, cells=cells, tolerance=tol)


def _kernel(path: Path) -> dict:
    out: dict = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        f = line.split()
        if not f or f[0] == "identity":
            continue
        if f[0] == "lengthScale":
            out["length_scales"] = [float(x) for x in f[2:]]
        elif len(f) >= _MIN_TOKENS:
            try:
                out[f[0]] = float(f[1])
            except ValueError:
                pass
    return out


def _search_from_log(log: dict[str, list[dict]], tol, converged) -> SearchHistory | None:
    rows: dict[int, dict] = {}
    for d in log.get("gp_oie", []):
        if "max_gp_F" in d and "outer" in d:
            rows[int(d["outer"])] = d
    if not rows:
        return None
    outers = sorted(rows)
    col = lambda k: np.array([rows[o].get(k, np.nan) for o in outers], dtype=float)  # noqa: E731
    series = {
        "outer": np.array(outers, dtype=float),
        "oracle_calls": col("oracle_calls"),
        "true_force": col("max_oracle_F"),
        "ci_true_force": col("ci_true_Fmax"),
        "surrogate_force": col("max_gp_F"),
        "ci_surrogate_force": col("ci_gp_F"),
        "max_variance": col("max_var"),
        "n_train": col("ntrain"),
        "magnitude_sigma2": col("mag2"),
        "noise_sigma2": col("s2n"),
        "length_scale_min": col("lmin"),
        "length_scale_max": col("lmax"),
        "time_refit": col("pt_refit"),
        "time_predict": col("pt_pred"),
        "time_inner": col("pt_inner"),
        "rss_mb": col("rss_mb"),
    }
    series = {k: v for k, v in series.items() if np.isfinite(v).any()}
    calls_at = dict(zip(outers, series["oracle_calls"], strict=True))
    events = []
    for d in log.get("acq-pick", []):
        o = int(d.get("outer", -1))
        events.append(
            AcquisitionEvent(
                oracle_calls=int(calls_at.get(o, -1)),
                kind=str(d.get("arm", "pick")),
                outer=o,
                image=int(d.get("worst", -1)),
            )
        )
    for d in log.get("oie-measured-step", []):
        o = int(d.get("outer", -1))
        events.append(
            AcquisitionEvent(
                oracle_calls=int(calls_at.get(o, -1)),
                kind="measured-step",
                outer=o,
                image=int(d.get("image", -1)),
                detail=" ".join(d.get("flags", [])),
            )
        )
    cap = None
    for d in log.get("ceiling", []):
        events.append(
            AcquisitionEvent(
                oracle_calls=int(calls_at.get(int(d.get("outer", -1)), -1)),
                kind="ceiling",
                outer=int(d.get("outer", -1)),
                detail=f"dropped={int(d.get('dropped', 0))}",
            )
        )
        cap = int(d.get("ntrain", 0)) or cap
    return SearchHistory(
        series=series,
        force_tolerance=tol,
        converged=converged,
        events=events,
        retained_cap=cap,
    )


def _projection(points: np.ndarray, path: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """Project rows of ``points`` onto the polyline ``path``; return path coordinate."""
    out = np.empty(len(points))
    for i, p in enumerate(points):
        best, bs = np.inf, 0.0
        for j in range(len(path) - 1):
            a, b = path[j], path[j + 1]
            ab = b - a
            den = float(ab @ ab)
            t = 0.0 if den == 0 else float(np.clip((p - a) @ ab / den, 0.0, 1.0))
            d = float(np.linalg.norm(p - (a + t * ab)))
            if d < best:
                best, bs = d, coord[j] + t * (coord[j + 1] - coord[j])
        out[i] = bs
    return out


def _band_from_h5(h5_path: Path, events) -> BandHistory:
    import h5py

    with h5py.File(h5_path, "r") as f:
        keys = sorted(f["converged"].keys(), key=int)
        energy = np.array([f["converged"][k]["energy"][0] for k in keys])
        pos = np.array([f["converged"][k]["positions"][:] for k in keys])
        ci = (
            int(f["summary/max_energy_image"][0])
            if "summary/max_energy_image" in f
            else None
        )
        pts = None
        if "training" in f:
            nr = int(f["training/n_rows"][0])
            dof = int(f["training/dof"][0])
            tp = f["training/positions"][:].reshape(-1, dof)[:nr]
            te = f["training/energies"][:nr]
            tg = f["training/gradients"][:].reshape(-1, dof)[:nr]
            pts = (tp, te, tg, dof)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    coord = np.concatenate([[0.0], np.cumsum(step)])
    true_e = np.full(len(keys), np.nan)
    true_f = np.full(len(keys), np.nan)
    points = None
    if pts is not None:
        tp, te, tg, dof = pts
        atoms = dof // 3
        fmax = np.linalg.norm(tg.reshape(-1, atoms, 3), axis=2).max(axis=1)
        d = np.linalg.norm(tp[:, None, :] - pos[None], axis=2)
        nearest = d.argmin(axis=0)
        hit = d.min(axis=0) < MATCH_TOLERANCE
        true_e[hit] = te[nearest[hit]]
        true_f[hit] = fmax[nearest[hit]]
        points = EvaluatedPoints(
            energy=te,
            max_force=fmax,
            positions=tp,
            coordinate=_projection(tp, pos, coord),
        )
    final = BandSnapshot(
        coordinate=coord,
        energy=energy,
        true_energy=true_e,
        true_force=true_f,
        climbing=ci,
        positions=pos,
    )
    return BandHistory(final=final, points=points, events=events)


def _dimer_from_mlflow(run: Path, tol, converged) -> SingleEndedHistory | None:
    m = read_mlflow_metrics(run)
    if "dimer.oracle_calls" not in m or "dimer.force" not in m:
        return None
    calls = m["dimer.oracle_calls"]
    n = len(calls["value"])
    force = m["dimer.force"]["value"][:n]
    curv = m.get("dimer.curvature", {"value": np.full(n, np.nan)})["value"][:n]
    t_row = calls["time_ms"]

    def paired(name):
        out = np.full(n, np.nan)
        if name not in m:
            return out
        for t, v in zip(m[name]["time_ms"], m[name]["value"], strict=True):
            j = int(np.argmin(np.abs(t_row - t)))
            if abs(t_row[j] - t) < PAIR_WINDOW_MS:
                out[j] = v
        return out

    escape = []
    cv = calls["value"]
    for j in range(1, n):
        if cv[j] - cv[j - 1] >= SPECTRUM_BATCH_CALLS:
            escape.append(
                AcquisitionEvent(
                    oracle_calls=int(cv[j]),
                    kind="spectrum",
                    outer=int(calls["step"][j]),
                    detail=f"{int(cv[j] - cv[j - 1])} calls",
                )
            )
    return SingleEndedHistory(
        oracle_calls=cv,
        force=force,
        curvature=curv,
        curvature_measured=paired("dimer.measured_curvature"),
        curvature_surrogate=paired("dimer.surrogate_curvature"),
        escape=escape,
        force_tolerance=tol,
        converged=converged,
    )


def parse_gpr_optim_cell(
    cell_dir: str | Path, *, mlruns: str | Path | None = None
) -> SurrogateSearch:
    """Parse one benchmark cell into a :class:`SurrogateSearch`.

    Parameters
    ----------
    cell_dir
        ``<record>/<set>/<cell>``.
    mlruns
        MLflow store root; defaults to ``<record>/mlruns``.
    """
    cell = Path(cell_dir)
    rec = parse_gpr_optim_result(cell / "result.json")
    tol = rec.extras.get("force_tolerance")
    mlruns = Path(mlruns) if mlruns else None
    band_log = cell / "band" / "gprn.log"
    rec.ledger = _ledger(band_log)
    _add_dimer_calls(rec, cell)
    prov = {"result.json": sha256_file(cell / "result.json")}
    search = band = single = None
    if band_log.is_file():
        log = parse_gprn_log(band_log)
        search = _search_from_log(log, tol, rec.converged)
        prov["band/gprn.log"] = sha256_file(band_log)
        if search is not None:
            search.meta["kernel"] = _kernel(cell / "band" / "obs-archive" / "kernel.txt")
            search.meta["total_calls"] = rec.search_calls
        h5 = cell / "band" / "band.h5"
        if h5.is_file():
            band = _band_from_h5(h5, search.events if search else [])
            prov["band/band.h5"] = sha256_file(h5)
    run = _locate_run(cell, "saddle", mlruns)
    if run is not None:
        single = _dimer_from_mlflow(run, tol, rec.converged)
    return SurrogateSearch(
        label=rec.label,
        producer="gpr_optim",
        band=band,
        search=search,
        single_ended=single,
        cell=rec,
        provenance=prov,
    )
