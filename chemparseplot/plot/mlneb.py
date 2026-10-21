"""Force and path-uncertainty histories for unfinished ML-NEB searches."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_mlneb_diagnostics(history, summaries, titles=None, *, width=7.2):
    """Show measured selected-point forces separately from energy uncertainty."""
    cases = [r for r in summaries if str(r["accepted"]).lower() == "false"]
    if not cases:
        msg = "No unsuccessful searches in the diagnostic input"
        raise ValueError(msg)
    titles = titles or {}
    fig, axes = plt.subplots(
        (len(cases) + 1) // 2,
        2,
        figsize=(width, 2.35 * ((len(cases) + 1) // 2)),
        squeeze=False,
        layout="constrained",
    )
    for index, (ax, case) in enumerate(zip(axes.flat, cases, strict=False)):
        rows = [r for r in history if r["label"] == case["label"] and int(r["step"]) > 0]
        n = np.array([int(r["observations"]) for r in rows])
        f = np.array(
            [float(r["sample_fmax_eV_A"]) / float(r["force_target_eV_A"]) for r in rows]
        )
        u = np.array(
            [
                float(r["max_uncertainty_eV"]) / float(r["uncertainty_target_eV"])
                for r in rows
            ]
        )
        ax.plot(n, f, color="#0072b2", linewidth=1.3)
        ax.plot(n, np.where(u > 0, u, np.nan), color="#cc79a7", linewidth=1.3)
        ax.axhline(1, color="#444444", linestyle="--", linewidth=0.8)
        ax.set_yscale("log")
        positive = np.concatenate((f[f > 0], u[u > 0]))
        ax.set_ylim(min(0.1, positive.min() / 2), max(300, positive.max() * 2))
        name = titles.get(case["label"], case["label"].replace("_", " "))
        ax.set_title(
            f"{'abcdefghijklmnopqrstuvwxyz'[index]}) {name}\n"
            f"CatLearn: {int(case['observations'])} calls, "
            f"{float(case['search_wall_s']):.0f} s (timeout)\n"
            f"GPRidge: {int(case['gp_search_calls'])} calls, "
            f"{float(case['gp_search_wall_s']):.1f} s",
            loc="left", fontsize=8,
        )
        ax.set_xlabel("Energy and force observations")
        if index % 2 == 0:
            ax.set_ylabel("Value / stopping threshold")
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(cases) :]:
        ax.set_visible(False)
    fig.legend(
        handles=[
            Line2D([], [], color="#0072b2", label="True force at selected point"),
            Line2D([], [], color="#cc79a7", label="Path energy uncertainty"),
            Line2D([], [], color="#444444", linestyle="--", label="Stopping threshold"),
        ],
        loc="outside lower center",
        ncols=2,
        frameon=False,
        fontsize=8,
    )
    return fig
