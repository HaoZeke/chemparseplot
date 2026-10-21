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
        msg = "Duplicate reaction in performance profile"
        raise ValueError(msg)
    labels = sorted({r["label"] for r in selected})
    if not labels or any((m, label) not in group for m in methods for label in labels):
        msg = "Methods must contain the same complete reaction set"
        raise ValueError(msg)
    ratios = {m: [] for m in methods}
    for label in labels:
        costs = {}
        for method in methods:
            row = group[method, label]
            if row["accepted"]:
                value = row[metric]
                if not math.isfinite(value) or value <= 0:
                    msg = "Accepted performance costs must be finite and positive"
                    raise ValueError(msg)
                costs[method] = value
        best = min(costs.values(), default=None)
        for method in methods:
            ratios[method].append(costs[method] / best if method in costs else None)
    return {"labels": labels, "ratios": ratios}


def cost_frontier(rows, panel, methods, *, exclude=("plain",)):
    """Mean costs on the same successful cases, retaining full-panel acceptance.

    Equal points both belong to the frontier. A method dominates another only
    when neither cost is larger and at least one is strictly smaller.
    """
    checked = performance_ratios(rows, panel, methods, "search_calls")
    performance_ratios(rows, panel, methods, "search_wall_s")
    labels = checked["labels"]
    group = {(r["method"], r["label"]): r for r in rows if r["set"] == panel}
    locators = [m for m in methods if m not in exclude]
    common = [k for k in labels if all(group[m, k]["accepted"] for m in locators)]
    points = []
    if locators and common:
        for method in locators:
            points.append({
                "method": method,
                **{f"mean_{key}": sum(group[method, k][key] for k in common) / len(common)
                   for key in ("search_calls", "search_wall_s")},
            })
        for point in points:
            point["nondominated"] = not any(
                all(other[key] <= point[key] for key in ("mean_search_calls", "mean_search_wall_s"))
                and any(other[key] < point[key] for key in ("mean_search_calls", "mean_search_wall_s"))
                for other in points
            )
    return {
        "common_accepted_labels": common if locators else [],
        "points": points,
        "acceptance": {m: sum(group[m, k]["accepted"] for k in labels) for m in methods},
        "cases": len(labels),
    }
