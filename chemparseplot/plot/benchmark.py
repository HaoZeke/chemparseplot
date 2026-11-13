"""Direct cost comparisons with full-panel failures kept visible."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

from chemparseplot.parse.benchmark import paired_costs

TEXT_CONTRAST_THRESHOLD = 2.7
MAJOR_TICKS_ONLY_DECADES = 1.5
GP_COLOR = "#008577"
BASE_COLOR = "#69778b"

METHOD_COLORS = {
    "gpridge": GP_COLOR,
    "plain": "#999999",
    "cineb": "#0072b2",
    "ocineb": "#56b4e9",
    "catlearn": "#cc79a7",
    "mlfsm": "#d55e00",
    "dimer": "#e69f00",
    "sella": "#785ef0",
}


def _plain_log_ticks(axis, lo, hi):
    """Label log-axis majors as plain numbers and leave minors unlabelled."""
    decades = np.log10(hi / lo)
    if decades > MAJOR_TICKS_ONLY_DECADES:
        subs = (1,)
    elif decades > 1:
        subs = (1, 2, 5)
    else:
        subs = tuple(range(1, 10))
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


def _sparse_log_ticks(axis, lo, hi, minimum=3):
    """Label the sparsest 1-2-5 style series with at least ``minimum`` ticks."""
    for subs in ((1,), (1, 2, 5), (1, 2, 3, 5), tuple(range(1, 10))):
        decades = range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 2)
        ticks = [m * 10.0**d for d in decades for m in subs if lo <= m * 10.0**d <= hi]
        if len(ticks) >= minimum:
            break
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    axis.set_minor_locator(LogLocator(base=10, subs=tuple(range(1, 10))))
    axis.set_minor_formatter(NullFormatter())


METHOD_MARKERS = {
    "gpridge": "o",
    "cineb": "s",
    "ocineb": "^",
    "catlearn": "D",
    "mlfsm": "v",
    "dimer": "P",
    "sella": "X",
    "plain": "h",
}
LIMIT_TOLERANCE = 1e-9  # decades; where the fitted log limits stop moving
PAIRED_METRICS = (
    ("search_calls", "Mean search force calls"),
    ("search_wall_s", "Mean search time (s)"),
)


def _text_width(text, size, weight="normal"):
    """Width in points of ``text`` in the current font family."""
    from matplotlib.font_manager import FontProperties  # noqa: PLC0415
    from matplotlib.textpath import TextPath  # noqa: PLC0415

    prop = FontProperties(weight=weight)
    return float(TextPath((0, 0), text, size=size, prop=prop).get_extents().width)


def _log_limits_with_room(pairs, room, width):
    """Log-axis limits that leave ``room`` points beside each pair of values.

    ``pairs`` holds the ``(low, high)`` values of each lane and ``room`` the
    points needed left of the low value and right of the high value. The
    limits are the fixed point at which every lane has exactly the room it
    asks for or more on an axis ``width`` points long.
    """
    logs = [(np.log10(a), np.log10(b)) for a, b in pairs]
    lo = min(a for a, _ in logs)
    hi = max(b for _, b in logs)
    span = max(hi - lo, 0.05)
    lo, hi = lo - 0.02 * span, hi + 0.02 * span
    for _ in range(200):
        span = hi - lo
        new_lo = min(
            a - left / width * span for (a, _), (left, _) in zip(logs, room, strict=True)
        )
        new_hi = max(
            b + right / width * span
            for (_, b), (_, right) in zip(logs, room, strict=True)
        )
        if max(abs(new_lo - lo), abs(new_hi - hi)) < LIMIT_TOLERANCE:
            break
        lo, hi = new_lo, new_hi
    return 10**lo, 10**hi


def plot_paired_means(
    reports,
    panels,
    methods,
    *,
    width=7.2,
    metrics=PAIRED_METRICS,
    reference_label="GPRidge",
    reference_color=GP_COLOR,
    method_color=BASE_COLOR,
    method_text_color=None,
    markers=None,
    show_ratio=True,
    lane=22.0,
    title_size=9.0,
    label_size=8.0,
    value_size=8.0,
    tick_size=7.5,
):
    """Paired mean costs of each method and the reference, one lane per method.

    ``reports`` maps a panel to ``{method: paired_costs(...)}``, the reports
    :func:`plot_cost_savings` returns and the comparison manifest stores.
    Each lane joins the mean of the method and the mean of the reference on
    the reactions both accept and prints each mean on the outer side of its
    marker, so two close means never share a spot. The lane labels are
    written once, left of the first column, and the axis limits are fitted
    to the printed values so the plot area carries no empty margin.

    Sizes are in points at the given ``width``; include the figure at that
    width to keep them. ``show_ratio`` prints the method total divided by the
    reference total at the end of each lane.
    """
    markers = {**METHOD_MARKERS, **(markers or {})}
    method_text_color = method_text_color or method_color
    total = width * 72
    lead = 1.3
    lanes = {
        panel: [m for m in methods if m != "plain" and m in reports[panel]]
        for panel in panels
    }

    def lane_label(panel, method):
        s = reports[panel][method]
        return f"{methods[method]}  {s['accepted']}/{s['total']}"

    def ratio_text(value):
        return f"{value:.3g}\u00d7"

    labels_w = max(
        _text_width(lane_label(panel, m), label_size)
        for panel in panels
        for m in lanes[panel]
    )
    ratio_w = (
        max(_text_width(ratio_text(999.0), value_size, "bold"), 1.0) + 6.0
        if show_ratio
        else 0.0
    )
    side, gap, between = 2.0, 8.0, 16.0
    left = side + labels_w + gap
    plot_w = (
        total - left - side - len(metrics) * ratio_w - (len(metrics) - 1) * between
    ) / len(metrics)
    title_h = lead * title_size + 3.0
    axis_h = 2.5 + 2.0 + lead * tick_size + 2.0 + lead * label_size + 4.0
    key_h = lead * label_size + 8.0
    heights = {panel: len(lanes[panel]) * lane for panel in panels}
    height = sum(title_h + heights[p] + axis_h for p in panels) + key_h
    fig = plt.figure(figsize=(width, height / 72))
    marker_pt = 6.5
    reach = marker_pt / 2 + 3.0
    y = height
    letter = 0
    for panel, title in panels.items():
        y -= title_h
        rows = lanes[panel]
        for col, (metric, xlabel) in enumerate(metrics):
            x0 = left + col * (plot_w + ratio_w + between)
            fig.text(
                x0 / total,
                (y + 0.45 * title_size) / height,
                f"({'abcdefghijklmnopqrstuvwxyz'[letter]}) {title}",
                fontsize=title_size,
                fontweight="bold",
                va="baseline",
            )
            letter += 1
            ax = fig.add_axes(
                (
                    x0 / total,
                    (y - heights[panel]) / height,
                    plot_w / total,
                    heights[panel] / height,
                )
            )
            pairs, room, drawn = [], [], []
            for lane_index, m in enumerate(rows):
                s = reports[panel][m]
                n = len(s["labels"])
                if not n:
                    ax.text(
                        0.02,
                        lane_index,
                        "No shared successes",
                        transform=ax.get_yaxis_transform(),
                        fontsize=value_size,
                        va="center",
                    )
                    continue
                a, b = s[metric]["method_total"] / n, s[metric]["reference_total"] / n
                (lo, lo_text, lo_color), (hi, hi_text, hi_color) = sorted(
                    [
                        (a, f"{a:.3g}", method_text_color),
                        (b, f"{b:.3g}", reference_color),
                    ],
                    key=lambda item: item[0],
                )
                pairs.append((lo, hi))
                room.append(
                    (
                        reach + _text_width(lo_text, value_size) + 2.0,
                        reach + _text_width(hi_text, value_size) + 2.0,
                    )
                )
                drawn.append((lane_index, m, a, b, s[metric]["ratio"]))
                ax.plot(
                    [a, b], [lane_index] * 2, color="#C9CED3", linewidth=1.2, zorder=1
                )
                ax.plot(
                    [a],
                    [lane_index],
                    marker=markers.get(m, "s"),
                    markersize=marker_pt,
                    color=method_color,
                    markeredgewidth=0,
                    linestyle="none",
                    zorder=3,
                )
                ax.plot(
                    [b],
                    [lane_index],
                    marker=markers["gpridge"],
                    markersize=marker_pt,
                    color=reference_color,
                    markeredgewidth=0,
                    linestyle="none",
                    zorder=3,
                )
                for value, text, color, direction in (
                    (lo, lo_text, lo_color, -1),
                    (hi, hi_text, hi_color, 1),
                ):
                    ax.annotate(
                        text,
                        (value, lane_index),
                        xytext=(direction * reach, 0),
                        textcoords="offset points",
                        ha="right" if direction < 0 else "left",
                        va="center_baseline",
                        fontsize=value_size,
                        color=color,
                    )
            ax.set_xscale("log")
            if pairs:
                ax.set_xlim(*_log_limits_with_room(pairs, room, plot_w))
            ax.set_ylim(len(rows) - 0.5, -0.5)
            _sparse_log_ticks(ax.xaxis, *ax.get_xlim())
            ax.set_yticks(range(len(rows)))
            if col == 0:
                ax.set_yticklabels(
                    [lane_label(panel, m) for m in rows], fontsize=label_size
                )
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0, pad=gap)
            ax.tick_params(axis="x", labelsize=tick_size, length=2.5, width=0.6, pad=2.0)
            ax.tick_params(axis="x", which="minor", length=1.5, width=0.4)
            ax.set_xlabel(xlabel, fontsize=label_size, labelpad=2.0)
            ax.grid(axis="x", which="major", color="#E3E6E8", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.6)
            if show_ratio:
                for lane_index, _m, _a, _b, ratio in drawn:
                    ax.text(
                        1.0 + 6.0 / plot_w,
                        lane_index,
                        ratio_text(ratio),
                        transform=ax.get_yaxis_transform(),
                        va="center_baseline",
                        fontsize=value_size,
                        fontweight="bold",
                        color=reference_color if ratio >= 1 else method_text_color,
                    )
        y -= heights[panel] + axis_h
    # Key: the reference marker, then every comparator marker before one label.
    present = list(dict.fromkeys(m for panel in panels for m in lanes[panel]))
    overlay = fig.add_axes((0, 0, 1, 1), zorder=-1)
    overlay.set_axis_off()
    overlay.set_xlim(0, total)
    overlay.set_ylim(0, height)
    note = "comparator, named on its row"
    step = marker_pt + 3.0
    key_w = (
        step
        + _text_width(reference_label, label_size)
        + 18.0
        + len(present) * step
        + _text_width(note, label_size)
    )
    x = (total - key_w) / 2
    mid = 0.5 * key_h
    baseline = mid - 0.36 * label_size

    def mark(x, marker, color):
        overlay.plot(
            [x + marker_pt / 2],
            [mid],
            marker=marker,
            markersize=marker_pt,
            color=color,
            markeredgewidth=0,
            linestyle="none",
            clip_on=False,
        )

    mark(x, markers["gpridge"], reference_color)
    x += step
    fig.text(
        x / total, baseline / height, reference_label, fontsize=label_size, va="baseline"
    )
    x += _text_width(reference_label, label_size) + 18.0
    for m in present:
        mark(x, markers.get(m, "s"), method_color)
        x += step
    fig.text(x / total, baseline / height, note, fontsize=label_size, va="baseline")
    return fig


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
        reference = [r for r in rows if r["set"] == panel and r["method"] == "gpridge"]
        check_label = (
            f"GPRidge: {sum(r['accepted'] for r in reference)}/{len(reference)} "
            "index-1 checks"
        )
        for j, (metric, xlabel) in enumerate(
            (
                ("search_calls", "Mean search force calls"),
                ("search_wall_s", "Mean search time (s)"),
            )
        ):
            ax = axes[i, j]
            ax.set_title(
                f"{'abcd'[i * 2 + j]}) {title}\n{check_label}",
                loc="left",
                fontweight="bold",
                fontsize=9,
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
                text = f"{ratio:.3g}\u00d7"
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
            ax.margins(x=0.25)
            _plain_log_ticks(ax.xaxis, *ax.get_xlim())
            ax.tick_params(axis="x", labelsize=8)
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


def plot_performance_profiles(rows, panels, methods, *, width=7.2):
    """Dolan-More profiles over every reaction, with failure assigned infinite cost."""
    from chemparseplot.parse.benchmark import performance_ratios  # noqa: PLC0415

    labels = {"gpridge": "GPRidge", **methods}
    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=(width, 2.5 * len(panels)),
        squeeze=False,
        layout="constrained",
    )
    reports, handles = {}, {}
    for i, (panel, title) in enumerate(panels.items()):
        present = [
            m for m in labels if any(r["set"] == panel and r["method"] == m for r in rows)
        ]
        reports[panel] = {}
        for j, (metric, name) in enumerate(
            (("search_calls", "Force calls"), ("search_wall_s", "Search time"))
        ):
            ax = axes[i, j]
            report = performance_ratios(rows, panel, present, metric)
            reports[panel][metric] = report
            finite = [
                v for values in report["ratios"].values() for v in values if v is not None
            ]
            upper = max(2, max(finite, default=1) * 1.15)
            x = np.array(sorted({1.0, upper, *finite}))
            for method in present:
                values = np.array(
                    [np.inf if v is None else v for v in report["ratios"][method]]
                )
                fraction = (values[:, None] <= x).mean(axis=0)
                (line,) = ax.step(
                    x,
                    fraction,
                    where="post",
                    label=labels[method],
                    color=METHOD_COLORS.get(method),
                    linewidth=2.3 if method == "gpridge" else 1.5,
                    linestyle="--" if method == "plain" else "-",
                    zorder=4 if method == "gpridge" else 2,
                )
                handles[method] = line
            ax.set_xscale("log", base=2)
            ax.set_xlim(1, upper)
            ax.set_ylim(-0.025, 1.04)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
            ax.set_title(
                f"{'abcdefghijklmnopqrstuvwxyz'[2 * i + j]}) {title}: {name}",
                loc="left",
                fontsize=9,
                fontweight="bold",
            )
            ax.set_xlabel("Cost / best accepted cost")
            ax.set_ylabel("Fraction of all reactions")
            ax.grid(alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=list(handles.values()),
        loc="outside lower center",
        ncols=4,
        fontsize=8,
        frameon=False,
    )
    return fig, reports


def plot_cost_frontiers(rows, panels, methods, *, width=7.2):
    """Plot common-case Pareto means and acceptance on the full panels."""
    from chemparseplot.parse.benchmark import cost_frontier  # noqa: PLC0415

    labels = {"gpridge": "GPRidge", **methods}
    fig, axes = plt.subplots(
        2,
        len(panels),
        figsize=(width, 6.4),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    reports = {}
    markers = {
        "gpridge": "o",
        "cineb": "s",
        "ocineb": "D",
        "catlearn": "^",
        "mlfsm": "P",
        "dimer": "s",
        "sella": "^",
    }
    for col, (panel, title) in enumerate(panels.items()):
        present = [
            m for m in labels if any(r["set"] == panel and r["method"] == m for r in rows)
        ]
        summary = cost_frontier(rows, panel, present)
        reports[panel] = summary
        ax = axes[0, col]
        ax.set_title(
            f"{'abcdefghijklmnopqrstuvwxyz'[col]}) {title}\n"
            f"{len(summary['common_accepted_labels'])} common accepted reactions",
            fontsize=9,
        )
        for point in summary["points"]:
            method = point["method"]
            x, y = point["mean_search_calls"], point["mean_search_wall_s"]
            ax.scatter(
                x,
                y,
                s=48,
                color=METHOD_COLORS.get(method),
                marker=markers.get(method, "o"),
                edgecolors="black" if point["nondominated"] else "none",
                linewidths=1,
                zorder=3,
            )
            offset = (5, -13) if method == "ocineb" else (5, 9)
            ax.annotate(
                labels[method],
                (x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
            )
        front = sorted(
            (p for p in summary["points"] if p["nondominated"]),
            key=lambda p: p["mean_search_calls"],
        )
        if len(front) > 1:
            ax.plot(
                [p["mean_search_calls"] for p in front],
                [p["mean_search_wall_s"] for p in front],
                color="#444444",
                linestyle="--",
                linewidth=1,
                zorder=1,
            )
        if not summary["points"]:
            ax.text(
                0.5,
                0.5,
                "No common accepted reactions",
                transform=ax.transAxes,
                ha="center",
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Mean search force calls",
            ylabel="Mean search time (s)",
        )
        ax.margins(x=0.38, y=0.4)
        _plain_log_ticks(ax.xaxis, *ax.get_xlim())
        _plain_log_ticks(ax.yaxis, *ax.get_ylim())
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax = axes[1, col]
        total = summary["cases"]
        for i, (method, count) in enumerate(summary["acceptance"].items()):
            ax.barh(i, count, color=METHOD_COLORS.get(method), height=0.67)
            ax.text(count + 0.25, i, f"{count}/{total}", va="center", fontsize=8)
        ax.set_yticks(range(len(present)), [labels[m] for m in present])
        ax.invert_yaxis()
        ax.set_xlim(0, total * 1.22)
        ax.set_xlabel("Accepted saddles in the full panel")
        ax.set_title(
            f"{'abcdefghijklmnopqrstuvwxyz'[len(panels) + col]}) "
            f"{title}: all {total} reactions",
            fontsize=9,
        )
        ax.spines[["top", "right"]].set_visible(False)
    return fig, reports


def plot_acceptance_curves(rows, panel, title, methods, *, width=7.2):
    """Cactus-style cumulative acceptance versus absolute search costs."""
    from chemparseplot.parse.benchmark import performance_ratios  # noqa: PLC0415

    labels = {"gpridge": "GPRidge", **methods}
    present = [
        m for m in labels if any(r["set"] == panel and r["method"] == m for r in rows)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(width, 3.2), layout="constrained")
    for i, (field, label) in enumerate(
        (("search_wall_s", "Search time (s)"), ("search_calls", "Search force calls"))
    ):
        report = performance_ratios(rows, panel, present, field)
        total = len(report["labels"])
        measured = [
            r[field]
            for r in rows
            if r["set"] == panel and r["method"] in present and r["accepted"]
        ]
        xmin, xmax = (min(measured) / 1.3, max(measured) * 1.3) if measured else (1, 10)
        ax = axes[i]
        for method in present:
            values = sorted(
                r[field]
                for r in rows
                if r["set"] == panel and r["method"] == method and r["accepted"]
            )
            ax.step(
                [xmin, *values, xmax],
                [0, *[k / total for k in range(1, len(values) + 1)], len(values) / total],
                where="post",
                color=METHOD_COLORS.get(method),
                label=labels[method],
                linewidth=2.3 if method == "gpridge" else 1.5,
                linestyle="--" if method == "plain" else "-",
            )
        ax.set(
            xscale="log",
            xlim=(xmin, xmax),
            ylim=(0, 1.04),
            xlabel=label,
            ylabel="Fraction of all reactions accepted",
        )
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_title(f"{'ab'[i]}) {title}: {label}", fontsize=9)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="outside lower center", ncols=3, frameon=False, fontsize=8
    )
    return fig
