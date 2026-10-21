"""Scientific accounting for paired costs and rejected reactions."""

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
        {"set": "b", "method": m, "label": k, "accepted": ok,
         "search_calls": c, "search_wall_s": t}
        for m, k, ok, c, t in [
            ("a", "one", True, 10, 2), ("a", "two", True, 100, 20),
            ("b", "one", True, 10, 2), ("b", "two", False, 1, 1),
            ("c", "one", True, 20, 2), ("c", "two", True, 1, 1),
            ("plain", "one", False, 1, 1), ("plain", "two", False, 1, 1),
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
