"""Scientific accounting for paired costs and rejected reactions."""

import pytest
from chemparseplot.parse.benchmark import paired_costs


def test_only_shared_successes_enter_both_totals():
    rows = [
        dict(set="b", method=m, label=k, accepted=ok, search_calls=c, search_wall_s=t)
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
        dict(
            set="b",
            method="gpridge",
            label="a",
            accepted=True,
            search_calls=2,
            search_wall_s=1,
        )
    ]
    with pytest.raises(ValueError, match="same complete"):
        paired_costs(rows, "b", "other")


def test_no_shared_successes_have_no_speedup():
    rows = [
        dict(set="b", method=m, label="a", accepted=ok, search_calls=2, search_wall_s=1)
        for m, ok in [("gpridge", True), ("other", False)]
    ]
    result = paired_costs(rows, "b", "other")
    assert result["search_calls"]["ratio"] is None
    assert result["labels"] == []
