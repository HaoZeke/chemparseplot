"""Direct cost comparisons with full-panel failures kept visible."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from chemparseplot.parse.benchmark import paired_costs

TEXT_CONTRAST_THRESHOLD = 2.7
GP_COLOR = "#008577"
BASE_COLOR = "#69778b"


def plot_cost_savings(rows, panels, methods, *, width=7.2):
    """Paired mean costs; each method uses its shared successes with GPRidge."""
    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=(width, 2.65 * len(panels)),
        squeeze=False,
        layout="constrained",
    )
    reports = {}
    for i, (panel, title) in enumerate(panels.items()):
        present = [
            m
            for m in methods
            if m != "plain" and any(r["set"] == panel and r["method"] == m for r in rows)
        ]
        stats = {m: paired_costs(rows, panel, m) for m in present}
        reports[panel] = stats
        for j, (metric, xlabel) in enumerate(
            (
                ("search_calls", "Mean search force calls"),
                ("search_wall_s", "Mean search time (s)"),
            )
        ):
            ax = axes[i, j]
            ax.set_title(
                f"{'abcd'[i * 2 + j]}) {title}", loc="left", fontweight="bold", fontsize=9
            )
            for y, m in enumerate(present):
                s = stats[m]
                n = len(s["labels"])
                if not n:
                    ax.text(
                        0.02, y, "No shared successes", transform=ax.get_yaxis_transform()
                    )
                    continue
                a, b = s[metric]["method_total"] / n, s[metric]["reference_total"] / n
                ax.plot([a, b], [y, y], color="#b8c0c8", linewidth=4, zorder=1)
                ax.scatter(a, y, marker="s", color=BASE_COLOR, s=42, zorder=3)
                ax.scatter(b, y, marker="o", color=GP_COLOR, s=48, zorder=3)
                ax.annotate(
                    f"{a:.3g}",
                    (a, y),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=BASE_COLOR,
                )
                ax.annotate(
                    f"{b:.3g}",
                    (b, y),
                    xytext=(0, -14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=GP_COLOR,
                )
                ratio = s[metric]["ratio"]
                text = f"{ratio:.1f}\u00d7" if ratio >= 1 else f"{ratio:.2f}\u00d7"
                ax.text(
                    1.03,
                    y,
                    text,
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    fontweight="bold",
                    color=GP_COLOR if ratio >= 1 else "#9e4151",
                )
            ax.set_yticks(
                range(len(present)),
                [
                    f"{methods[m]}  {stats[m]['accepted']}/{stats[m]['total']}"
                    for m in present
                ],
                fontsize=8,
            )
            ax.set_ylim(len(present) - 0.45, -0.6)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="x", labelsize=8)
            ax.margins(x=0.25)
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=0.15)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
    fig.legend(
        handles=[
            Line2D([], [], color=GP_COLOR, marker="o", linestyle="none", label="GPRidge"),
            Line2D(
                [], [], color=BASE_COLOR, marker="s", linestyle="none", label="Comparator"
            ),
        ],
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    return fig, reports


def plot_reaction_costs(rows, panel, title, methods, *, width=7.2):
    """Per-reaction comparator/reference ratios; crosses denote rejection."""
    group = {(r["method"], r["label"]): r for r in rows if r["set"] == panel}
    labels = sorted(label for method, label in group if method == "gpridge")
    present = [m for m in methods if any(method == m for method, _ in group)]
    fig, axes = plt.subplots(
        1, 2, figsize=(width, 0.29 * len(labels) + 1.6), layout="constrained"
    )
    for ax, metric, heading in zip(
        axes,
        ("search_calls", "search_wall_s"),
        ("a) Force calls", "b) Search time"),
        strict=True,
    ):
        values = np.full((len(labels), len(present)), np.nan)
        for i, label in enumerate(labels):
            for j, method in enumerate(present):
                row, ref = group[method, label], group["gpridge", label]
                if row["accepted"] and ref["accepted"]:
                    values[i, j] = np.log2(row[metric] / ref[metric])
        cmap = plt.get_cmap("BrBG").copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(0, -5, 5), aspect="auto")
        for i, _label in enumerate(labels):
            for j, _method in enumerate(present):
                v = values[i, j]
                ax.text(
                    j,
                    i,
                    "\u00d7" if np.isnan(v) else f"{2**v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9 if np.isnan(v) else 7,
                    fontweight="bold" if np.isnan(v) else "normal",
                    color="white"
                    if np.isfinite(v) and abs(v) > TEXT_CONTRAST_THRESHOLD
                    else "#222222",
                )
        ax.set_xticks(
            range(len(present)),
            [methods[m] for m in present],
            rotation=55,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(
            range(len(labels)),
            [s.replace("_", " ", 1).replace("_", " ") for s in labels]
            if ax is axes[0]
            else [],
            fontsize=8,
        )
        ax.set_title(heading, loc="left", fontweight="bold")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        fraction=0.025,
        pad=0.02,
        extend="both",
        ticks=[-4, -2, 0, 2, 4],
    )
    cbar.ax.set_xticklabels(["1/16", "1/4", "1", "4", "16"])
    cbar.set_label("Comparator / GPRidge cost   (>1: GPRidge uses less)")
    fig.suptitle(
        f"{title}   ·   \u00d7 fails the common acceptance rule",
        fontsize=10,
        fontweight="bold",
    )
    return fig
