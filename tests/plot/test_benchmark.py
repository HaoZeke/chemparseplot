"""Scientific accounting for paired costs and rejected reactions."""

import numpy as np
import pytest

from chemparseplot.parse.benchmark import paired_costs


def test_only_shared_successes_enter_both_totals():
    rows = [
        {
            "set": "b",
            "method": m,
            "label": k,
            "accepted": ok,
            "search_calls": c,
            "search_wall_s": t,
        }
        for m, k, ok, c, t in [
            ("gpridge", "a", True, 10, 2),
            ("gpridge", "b", True, 100, 20),
            ("other", "a", True, 40, 3),
            ("other", "b", False, 1000, 900),
        ]
    ]
    result = paired_costs(rows, "b", "other")
    assert result["labels"] == ["a"]
    assert result["accepted"] == 1
    assert result["total"] == 2
    assert result["search_calls"]["ratio"] == 4
    assert result["search_wall_s"]["ratio"] == 1.5


def test_missing_reaction_is_not_a_success_only_campaign():
    rows = [
        {
            "set": "b",
            "method": "gpridge",
            "label": "a",
            "accepted": True,
            "search_calls": 2,
            "search_wall_s": 1,
        }
    ]
    with pytest.raises(ValueError, match="same complete"):
        paired_costs(rows, "b", "other")


def test_no_shared_successes_have_no_speedup():
    rows = [
        {
            "set": "b",
            "method": m,
            "label": "a",
            "accepted": ok,
            "search_calls": 2,
            "search_wall_s": 1,
        }
        for m, ok in [("gpridge", True), ("other", False)]
    ]
    result = paired_costs(rows, "b", "other")
    assert result["search_calls"]["ratio"] is None
    assert result["labels"] == []


def test_performance_profile_keeps_failures_and_counts_ties():
    from chemparseplot.parse.benchmark import performance_ratios

    rows = [
        {"set": "b", "method": m, "label": label, "accepted": ok, "cost": cost}
        for m, label, ok, cost in [
            ("a", "one", True, 1),
            ("b", "one", True, 2),
            ("a", "two", True, 4),
            ("b", "two", True, 4),
            ("a", "three", False, 0.001),
            ("b", "three", True, 3),
            ("a", "four", False, 0),
            ("b", "four", False, 0),
        ]
    ]
    result = performance_ratios(rows, "b", ["a", "b"], "cost")
    assert result["labels"] == ["four", "one", "three", "two"]
    assert result["ratios"]["a"] == [None, 1, None, 1]
    assert result["ratios"]["b"] == [None, 2, 1, 1]
    assert sum(v == 1 for v in result["ratios"]["a"]) / len(result["labels"]) == 0.5
    assert (
        sum(v is not None for v in result["ratios"]["b"]) / len(result["labels"]) == 0.75
    )
    rows[0]["cost"] = 0
    with pytest.raises(ValueError, match="finite and positive"):
        performance_ratios(rows, "b", ["a", "b"], "cost")


def test_frontier_uses_the_same_cases_and_preserves_equal_points():
    from chemparseplot.parse.benchmark import cost_frontier

    rows = [
        {
            "set": "b",
            "method": m,
            "label": k,
            "accepted": ok,
            "search_calls": c,
            "search_wall_s": t,
        }
        for m, k, ok, c, t in [
            ("a", "one", True, 10, 2),
            ("a", "two", True, 100, 20),
            ("b", "one", True, 10, 2),
            ("b", "two", False, 1, 1),
            ("c", "one", True, 20, 2),
            ("c", "two", True, 1, 1),
            ("plain", "one", False, 1, 1),
            ("plain", "two", False, 1, 1),
        ]
    ]
    report = cost_frontier(rows, "b", ["a", "b", "c", "plain"])
    assert report["common_accepted_labels"] == ["one"]
    assert report["acceptance"] == {"a": 2, "b": 1, "c": 2, "plain": 0}
    assert report["cases"] == 2
    assert [p["mean_search_calls"] for p in report["points"]] == [10, 10, 20]
    assert [p["nondominated"] for p in report["points"]] == [True, True, False]
    rows[0]["accepted"] = False
    report = cost_frontier(rows, "b", ["a", "b", "c", "plain"])
    assert report["points"] == []
    assert report["common_accepted_labels"] == []


def _paired_reports():
    def stats(accepted, total, calls, walls):
        return {
            "labels": [f"case{i}" for i in range(accepted)],
            "accepted": accepted,
            "total": total,
            "search_calls": {
                "method_total": calls[0],
                "reference_total": calls[1],
                "ratio": calls[0] / calls[1],
            },
            "search_wall_s": {
                "method_total": walls[0],
                "reference_total": walls[1],
                "ratio": walls[0] / walls[1],
            },
        }

    return {
        "first": {
            "cineb": stats(18, 24, (5964.0, 767.0), (152.18, 78.18)),
            "catlearn": stats(18, 24, (860.0, 799.0), (6247.63, 69.84)),
        },
        "second": {"sella": stats(18, 20, (201.0, 672.0), (89.60, 229.37))},
    }


def test_log_limits_give_every_printed_value_its_room():
    from chemparseplot.plot.benchmark import _log_limits_with_room

    pairs = [(42.6, 331.0), (44.4, 47.8), (54.2, 291.0)]
    room = [(30.0, 28.0), (30.0, 30.0), (30.0, 28.0)]
    width = 170.0
    lo, hi = _log_limits_with_room(pairs, room, width)
    span = np.log10(hi / lo)
    slack = []
    for (a, b), (left, right) in zip(pairs, room, strict=True):
        slack.append(np.log10(a / lo) / span * width - left)
        slack.append(np.log10(hi / b) / span * width - right)
    assert min(slack) > -1e-6
    # the tightest lane on each side has no spare room, so the axis is not padded
    assert min(slack[0::2]) < 1e-6
    assert min(slack[1::2]) < 1e-6


def test_paired_means_print_each_mean_beside_its_marker():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from chemparseplot.plot.benchmark import plot_paired_means

    reports = _paired_reports()
    panels = {"first": "First", "second": "Second"}
    methods = {"cineb": "CI-NEB", "catlearn": "CatLearn", "sella": "Sella"}
    fig = plot_paired_means(reports, panels, methods, width=6.4, show_ratio=False)
    assert fig.get_figwidth() == 6.4
    axes = [ax for ax in fig.axes if ax.axison]
    assert len(axes) == 4
    calls, walls = axes[0], axes[1]
    printed = sorted(t.get_text() for t in calls.texts)
    assert printed == sorted(["331", "42.6", "47.8", "44.4"])
    assert sorted(t.get_text() for t in walls.texts) == sorted(
        ["8.45", "4.34", "347", "3.88"]
    )
    assert sorted(t.get_text() for t in axes[2].texts) == sorted(["11.2", "37.3"])
    # lane labels are written once, beside the first column
    assert [t.get_text() for t in calls.get_yticklabels()] == [
        "CI-NEB  18/24",
        "CatLearn  18/24",
    ]
    assert all(not t.get_text() for t in walls.get_yticklabels())
    assert not any("\u00d7" in t.get_text() for ax in axes for t in ax.texts)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        frame = ax.get_window_extent(renderer)
        boxes = [t.get_window_extent(renderer) for t in ax.texts]
        for box in boxes:
            assert frame.x0 - 1 <= box.x0 and box.x1 <= frame.x1 + 1
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                assert not boxes[a].overlaps(boxes[b])
        # no value sits on a marker of its axes
        for line in ax.lines:
            if line.get_marker() in ("None", None, ""):
                continue
            x, y = ax.transData.transform((line.get_xdata()[0], line.get_ydata()[0]))
            half = line.get_markersize() / 2 * fig.dpi / 72
            for box in boxes:
                assert not (
                    box.x0 < x + half and x - half < box.x1 and box.y0 < y < box.y1
                )
        # the fitted limits leave no decade of empty axis
        values = [line.get_xdata()[0] for line in ax.lines if len(line.get_xdata()) == 1]
        lo, hi = ax.get_xlim()
        assert np.log10(hi / lo) < np.log10(max(values) / min(values)) + 1.0
    sizes = [t.get_fontsize() for t in fig.findobj(mpl.text.Text) if t.get_text()]
    assert min(sizes) >= 7.5
    plt.close(fig)


def test_paired_means_can_print_the_multiplier():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from chemparseplot.plot.benchmark import plot_paired_means

    fig = plot_paired_means(
        _paired_reports(),
        {"first": "First"},
        {"cineb": "CI-NEB", "catlearn": "CatLearn"},
        width=6.4,
    )
    axes = [ax for ax in fig.axes if ax.axison]
    ratios = [t.get_text() for t in axes[0].texts if "\u00d7" in t.get_text()]
    assert ratios == ["7.78\u00d7", "1.08\u00d7"]
    assert [t.get_text() for t in axes[1].texts if "\u00d7" in t.get_text()] == [
        "1.95\u00d7",
        "89.5\u00d7",
    ]
    plt.close(fig)


def test_sparse_log_ticks_label_at_least_three_values():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from chemparseplot.plot.benchmark import _sparse_log_ticks

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    for limits, expected in (
        ((1.9, 15.0), [2.0, 5.0, 10.0]),
        ((30.0, 450.0), [50.0, 100.0, 200.0]),
        ((1.0, 5000.0), [1.0, 10.0, 100.0, 1000.0]),
    ):
        ax.set_xlim(*limits)
        _sparse_log_ticks(ax.xaxis, *limits)
        assert list(ax.xaxis.get_majorticklocs()) == pytest.approx(expected)
    plt.close(fig)
