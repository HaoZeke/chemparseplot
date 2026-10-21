"""Paired search costs and complete reaction outcomes from campaign tables."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def load_benchmark_exports(paths, panels):
    """Read verified cells.csv exports without silently dropping failed cases.

    ``panels`` maps each panel name to its expected number of reactions. Costs
    include initialization when the export provides that count. A repeated GP
    reference must agree, including its result hash and source revision.
    """
    records, references = {}, {}

    def flag(value):
        if value not in ("True", "False"):
            raise ValueError(f"Invalid acceptance: {value!r}")
        return value == "True"

    def number(value):
        if value in (None, ""):
            return None
        result = float(value)
        if not math.isfinite(result) or result < 0:
            msg = "Costs must be finite and nonnegative"
            raise ValueError(msg)
        return result

    for path in paths:
        with Path(path).open(newline="") as stream:
            for raw in csv.DictReader(stream):
                panel, method, label = raw["set"], raw["method"], raw["label"]
                if panel not in panels or method == "gpridge":
                    raise ValueError(f"Unexpected export group: {panel}/{method}")
                key = panel, method, label
                if key in records:
                    raise ValueError(f"Duplicate measurement: {key}")
                records[key] = {
                    "set": panel,
                    "method": method,
                    "label": label,
                    "accepted": flag(raw["accepted"]),
                    "search_calls": number(
                        raw.get("search_calls_with_initialization", raw["search_calls"])
                    ),
                    "search_wall_s": number(raw["search_wall_s"]),
                    "error": raw.get("error", ""),
                    "run_sha256": raw["run_sha256"],
                }
                reference = {
                    "set": panel,
                    "method": "gpridge",
                    "label": label,
                    "accepted": flag(raw["gp_accepted"]),
                    "search_calls": number(raw["gp_search_calls"]),
                    "search_wall_s": number(raw["gp_search_wall_s"]),
                    "run_sha256": raw["gp_result_sha256"],
                    "source_revision": raw["gp_revision"],
                }
                refkey = panel, "gpridge", label
                if refkey in references and references[refkey] != reference:
                    raise ValueError(f"Conflicting reference: {refkey}")
                references[refkey] = reference
    records.update(references)
    for panel, count in panels.items():
        expected = {label for p, m, label in references if p == panel}
        if len(expected) != count:
            raise ValueError(f"Incomplete reference panel: {panel}")
        for method in {m for p, m, _ in records if p == panel}:
            labels = {label for p, m, label in records if p == panel and m == method}
            if labels != expected:
                raise ValueError(f"Incomplete campaign: {panel}/{method}")
    for row in records.values():
        if row["accepted"] and any(
            row[k] is None or row[k] <= 0 for k in ("search_calls", "search_wall_s")
        ):
            msg = "Accepted searches require positive measured costs"
            raise ValueError(msg)
    return list(records.values())


def paired_costs(rows, panel, method, reference="gpridge"):
    """Ratio of total costs on the exact intersection of accepted reactions."""
    selected = {
        r["label"]: r for r in rows if r["set"] == panel and r["method"] == method
    }
    refs = {r["label"]: r for r in rows if r["set"] == panel and r["method"] == reference}
    if set(selected) != set(refs):
        msg = "Methods must contain the same complete reaction panel"
        raise ValueError(msg)
    labels = sorted(
        k for k in selected if selected[k]["accepted"] and refs[k]["accepted"]
    )
    result = {
        "labels": labels,
        "accepted": sum(r["accepted"] for r in selected.values()),
        "reference_accepted": sum(r["accepted"] for r in refs.values()),
        "total": len(refs),
    }
    for metric in ("search_calls", "search_wall_s"):
        a = sum(selected[k][metric] for k in labels)
        b = sum(refs[k][metric] for k in labels)
        result[metric] = {
            "method_total": a,
            "reference_total": b,
            "ratio": a / b if labels else None,
        }
    return result


def performance_ratios(rows, panel, methods, metric):
    """Return cost/best accepted cost; failed checks retain their denominator."""
    selected = [r for r in rows if r["set"] == panel and r["method"] in methods]
    group = {(r["method"], r["label"]): r for r in selected}
    if len(group) != len(selected):
        raise ValueError("Duplicate reaction in performance profile")
    labels = sorted({r["label"] for r in selected})
    if not labels or any((m, label) not in group for m in methods for label in labels):
        raise ValueError("Methods must contain the same complete reaction set")
    ratios = {m: [] for m in methods}
    for label in labels:
        costs = {}
        for method in methods:
            row = group[method, label]
            if row["accepted"]:
                value = row[metric]
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        "Accepted performance costs must be finite and positive"
                    )
                costs[method] = value
        best = min(costs.values(), default=None)
        for method in methods:
            ratios[method].append(costs[method] / best if method in costs else None)
    return {"labels": labels, "ratios": ratios}
