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


def project_on_path(
    points: np.ndarray, path: np.ndarray, coord: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project rows of ``points`` onto the polyline ``path``.

    Returns the path coordinate of the foot point and the Cartesian distance
    (norm over all coordinates, in the unit of ``path``) to it.
    """
    out = np.empty(len(points))
    dist = np.empty(len(points))
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
        out[i], dist[i] = bs, best
    return out, dist


def _projection(points: np.ndarray, path: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """Path coordinate of the foot point of each row of ``points``."""
    return project_on_path(points, path, coord)[0]


def _read_saddle(path: Path) -> np.ndarray | None:
    """Flat Cartesian positions (A) of an eOn ``.con`` file; None when unreadable."""
    if not path.is_file():
        return None
    try:
        from ase.io import read

        return np.asarray(read(path, format="eon").positions, dtype=float).ravel()
    except Exception:
        return None


def read_atom_types(path: str | Path, n_atoms: int | None = None) -> np.ndarray:
    """Atomic numbers of the first frame of a geometry file (ASE-readable; ``.con``
    files are read as eOn files).

    Raises ``ValueError`` naming the file when it cannot be read or when its atom
    count differs from ``n_atoms``.
    """
    from ase.io import read

    path = Path(path)
    try:
        atoms = read(path, index=0, format="eon" if path.suffix == ".con" else None)
    except Exception as exc:
        msg = f"cannot read atom types from {path}: {exc}"
        raise ValueError(msg) from exc
    if n_atoms is not None and len(atoms) != n_atoms:
        msg = f"{path} has {len(atoms)} atoms, the band has {n_atoms}"
        raise ValueError(msg)
    return np.asarray(atoms.numbers, dtype=int)


def _find_atom_types(cell: Path, band: BandHistory) -> None:
    """Fill ``band.numbers`` from the cell's own geometry files, in order."""
    n_atoms = band.final.positions.shape[1] // 3
    candidates = [cell / "band" / "band.con", cell / "saddle" / "pos.con"]
    band.numbers_looked_for = [str(c) for c in candidates]
    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            nums = read_atom_types(cand, n_atoms)
        except ValueError:
            continue
        band.numbers, band.numbers_source = nums, str(cand.relative_to(cell))
        return


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
        pc, pd = project_on_path(tp, pos, coord)
        points = EvaluatedPoints(
            energy=te,
            max_force=fmax,
            positions=tp,
            coordinate=pc,
            distance=pd,
            gradients=tg,
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
            _find_atom_types(cell, band)
            pos_con = cell / "saddle" / "pos.con"
            saddle = _read_saddle(pos_con)
            if saddle is not None:
                band.saddle = saddle
                band.saddle_certified = rec.passed
                prov["saddle/pos.con"] = sha256_file(pos_con)
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


def parse_scaling_csv(
    wall_csv: str | Path,
    counts_csv: str | Path | None = None,
    *,
    stage: str = "pipeline",
    cells: list[str] | None = None,
):
    """Strong-scaling table from per-repetition wall times.

    ``wall_csv`` has columns ``cell, ranks, threads, repetition, stage,
    seconds, calls``; ``counts_csv`` (optional) ``cell, ranks, threads,
    repetition, passed``. One series per cell, one point per
    ``ranks x threads`` layout (x = cores), time = median over repetitions
    with min and max kept, calls = median, and a point is capped when any of
    its repetitions has ``passed`` false.
    """
    import csv
    from collections import defaultdict

    from chemparseplot.parse.surrogate.model import ScalingTable

    passed: dict[tuple, bool] = {}
    if counts_csv:
        with Path(counts_csv).open() as fh:
            for r in csv.DictReader(fh):
                key = (r["cell"], int(r["ranks"]), int(r["threads"]))
                passed[key] = passed.get(key, True) and r["passed"] == "True"
    pts: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    search_calls: dict[tuple, list[float]] = defaultdict(list)
    with Path(wall_csv).open() as fh:
        for r in csv.DictReader(fh):
            if cells and r["cell"] not in cells:
                continue
            key = (int(r["ranks"]), int(r["threads"]))
            if r["stage"] == "search" and r["calls"]:
                search_calls[(r["cell"], *key)].append(float(r["calls"]))
            if r["stage"] != stage:
                continue
            calls = float(r["calls"]) if r["calls"] else np.nan
            pts[r["cell"]][key].append((float(r["seconds"]), calls))
    series, tmin, tmax, calls_d, capped, layouts = {}, {}, {}, {}, {}, {}
    for cell, by_layout in sorted(pts.items()):
        keys = sorted(by_layout, key=lambda k: (k[0] * k[1], k[1]))
        arr = [np.array(by_layout[k]) for k in keys]
        series[cell] = (
            np.array([k[0] * k[1] for k in keys], dtype=float),
            np.array([np.median(a[:, 0]) for a in arr]),
        )
        tmin[cell] = np.array([a[:, 0].min() for a in arr])
        tmax[cell] = np.array([a[:, 0].max() for a in arr])
        calls_d[cell] = np.array(
            [
                float(np.median(search_calls[(cell, *k)]))
                if search_calls.get((cell, *k))
                else np.nan
                for k in keys
            ]
        )
        layouts[cell] = [f"{k[0]}x{k[1]}" for k in keys]
        capped[cell] = np.array([not passed.get((cell, *k), True) for k in keys])
    return ScalingTable(
        series,
        time_min=tmin,
        time_max=tmax,
        calls=calls_d,
        capped=capped,
        layouts=layouts,
    )


def pop_csv_cells(path: str | Path) -> list[str] | None:
    """Distinct ``cell`` values of a fixed-work CSV, in file order.

    ``None`` when the file has no ``cell`` column: its rows then belong to
    whichever cell the caller draws.
    """
    import csv

    with Path(path).open() as fh:
        reader = csv.DictReader(fh)
        if "cell" not in (reader.fieldnames or []):
            return None
        return list(dict.fromkeys(r["cell"] for r in reader))


def _pop_rows(path, cell):
    """Rows of ``cell`` (every row without a cell column), as dicts.

    A named cell the file does not hold is an error that lists the cells it
    does hold, so a label that differs from the wall table cannot vanish.
    """
    import csv

    with Path(path).open() as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        rows = list(reader)
    if cell is None or "cell" not in fields:
        return fields, rows
    have = list(dict.fromkeys(r["cell"] for r in rows))
    if cell not in have:
        msg = f"{path} has no rows of cell {cell!r}; its cells are {have}"
        raise ValueError(msg)
    return fields, [r for r in rows if r["cell"] == cell]


def parse_pop_csv(
    path: str | Path, metrics: list[str], *, cell: str | None = None
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Per-layout means of efficiency columns from a fixed-work CSV.

    The file needs ``ranks`` and ``threads`` columns; each name in ``metrics``
    is a column of fractions. With ``cell`` given and a ``cell`` column
    present, only that cell's rows count, and a cell the file lacks is an
    error naming the cells it has. Rows of one layout are averaged; layouts
    sort by cores, then threads.
    """
    from collections import defaultdict

    fields, rows = _pop_rows(path, cell)
    missing = [m for m in metrics if m not in fields]
    if missing:
        msg = f"{path} has no column(s) {missing}"
        raise ValueError(msg)
    acc: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (int(r["ranks"]), int(r["threads"]))
        for m in metrics:
            if r[m] not in ("", "nan"):
                acc[key][m].append(float(r[m]))
    keys = sorted(acc, key=lambda k: (k[0] * k[1], k[1]))
    return (
        [f"{k[0]}x{k[1]}" for k in keys],
        {
            m: np.array([np.mean(acc[k][m]) if acc[k][m] else np.nan for k in keys])
            for m in metrics
        },
    )


def parse_pop_factors(
    path: str | Path,
    metrics: dict[str, str] | list[str],
    *,
    time: str = "elapsed_s",
    cell: str | None = None,
):
    """Fixed-work factors and wall per layout as a :class:`PopTable`.

    ``metrics`` maps a column of fractions to the label it is drawn with (a
    list uses the column names as labels); ``time`` names the column of the
    wall in seconds, from which speedup and fixed-work efficiency follow.
    Rows of one layout are averaged. The cell rule is that of
    :func:`parse_pop_csv`.
    """
    from chemparseplot.parse.surrogate.model import PopTable

    labels = dict(metrics) if isinstance(metrics, dict) else {m: m for m in metrics}
    layouts, values = parse_pop_csv(path, [*labels, time], cell=cell)
    cores = np.array([int(a) * int(b) for a, b in (lay.split("x") for lay in layouts)])
    return PopTable(
        layouts=layouts,
        cores=cores.astype(float),
        time=values.pop(time),
        metrics={labels[m]: v for m, v in values.items()},
        cell=cell,
    )


def parse_components_csv(
    path: str | Path,
    *,
    total: str = "total",
    cells: list[str] | None = None,
):
    """Median seconds per component from a tidy CSV as a :class:`ComponentTable`.

    Columns ``cell, ranks, threads, repetition, component, seconds``; the
    component named ``total`` is the whole wall of that run and every other
    component a non-nesting part of it. Repetitions of one cell and layout
    are reduced to their median. A requested cell the file lacks, and a run
    without a ``total`` row, are errors.
    """
    import csv
    from collections import defaultdict

    from chemparseplot.parse.surrogate.model import ComponentTable

    need = {"cell", "ranks", "threads", "repetition", "component", "seconds"}
    acc: dict[tuple, list[float]] = defaultdict(list)
    order: list[str] = []
    have: list[str] = []
    with Path(path).open() as fh:
        reader = csv.DictReader(fh)
        missing = need - set(reader.fieldnames or [])
        if missing:
            msg = f"{path} lacks column(s) {sorted(missing)}"
            raise ValueError(msg)
        for r in reader:
            if r["cell"] not in have:
                have.append(r["cell"])
            if cells and r["cell"] not in cells:
                continue
            comp = r["component"]
            if comp != total and comp not in order:
                order.append(comp)
            lay = f"{int(r['ranks'])}x{int(r['threads'])}"
            acc[(r["cell"], lay, comp)].append(float(r["seconds"]))
    if cells:
        absent = [c for c in cells if c not in have]
        if absent:
            msg = f"{path} has no rows of cell(s) {absent}; its cells are {have}"
            raise ValueError(msg)
    keys = sorted({(c, lay) for c, lay, _ in acc})
    parts: dict[tuple[str, str], dict[str, float]] = {}
    totals: dict[tuple[str, str], float] = {}
    layouts: dict[str, list[str]] = defaultdict(list)
    for cell, lay in keys:
        if (cell, lay, total) not in acc:
            msg = f"{path}: {cell} {lay} has no {total!r} component row"
            raise ValueError(msg)
        totals[(cell, lay)] = float(np.median(acc[(cell, lay, total)]))
        parts[(cell, lay)] = {
            c: float(np.median(acc[(cell, lay, c)]))
            for c in order
            if (cell, lay, c) in acc
        }
        layouts[cell].append(lay)

    def _cores(lay):
        a, b = lay.split("x")
        return int(a) * int(b), int(b)

    cell_list = cells or [c for c in have if c in layouts]
    return ComponentTable(
        components=order,
        cells=cell_list,
        layouts={c: sorted(layouts[c], key=_cores) for c in cell_list},
        parts=parts,
        total=totals,
    )


def parse_breakdown_csv(
    wall_csv: str | Path,
    components: list[str],
    *,
    total: str | None = "pipeline",
    cells: list[str] | None = None,
) -> dict[str, tuple[list[str], dict[str, np.ndarray], np.ndarray | None]]:
    """Median seconds of each component stage per cell and layout.

    Returns ``{cell: (layouts, {component: seconds}, total_seconds)}`` with
    layouts sorted by cores then threads. ``total`` names the stage holding
    the whole (``None`` for no total). The components should not nest.
    """
    import csv
    from collections import defaultdict

    wanted = {*components, *([total] if total else [])}
    acc: dict[tuple, list[float]] = defaultdict(list)
    keys: dict[str, set] = defaultdict(set)
    with Path(wall_csv).open() as fh:
        for r in csv.DictReader(fh):
            if r["stage"] not in wanted or (cells and r["cell"] not in cells):
                continue
            lay = (int(r["ranks"]), int(r["threads"]))
            keys[r["cell"]].add(lay)
            acc[(r["cell"], lay, r["stage"])].append(float(r["seconds"]))
    out = {}
    for cell, lays in sorted(keys.items()):
        order = sorted(lays, key=lambda k: (k[0] * k[1], k[1]))

        def med(stage, cell=cell, order=order):
            return np.array(
                [
                    float(np.median(acc[(cell, k, stage)]))
                    if acc.get((cell, k, stage))
                    else np.nan
                    for k in order
                ]
            )

        out[cell] = (
            [f"{k[0]}x{k[1]}" for k in order],
            {c: med(c) for c in components},
            med(total) if total else None,
        )
    return out
