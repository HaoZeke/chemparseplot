"""Scientific interpretation of repeated ML-NEB progress tables."""

from types import SimpleNamespace

import pytest

from chemparseplot.parse.mlneb import check_mlneb_forces, read_mlneb_history


def row(step, seconds, force):
    return (
        f"| ML-NEB | {step} | 2026-10-21 03:00:{seconds:02d} |"
        f" 1.0 | 2.0 | 0.1 | 0.05 | {force:.6f} |\n"
    )


def test_reprinted_timestamps_do_not_move_old_steps(tmp_path):
    path = tmp_path / "search.log"
    path.write_text(
        row(0, 1, 0.04)
        + row(0, 7, 0.04)
        + row(1, 7, 5.0)
        + row(0, 23, 0.04)
        + row(1, 23, 5.0)
        + row(2, 23, 0.3)
    )
    history = read_mlneb_history(path)
    assert [r["elapsed_s"] for r in history] == [0, 6, 22]
    assert [r["observations"] for r in history] == [2, 3, 4]
    assert [r["sample_fmax_eV_A"] for r in history] == [0.04, 5, 0.3]
    frames = [
        SimpleNamespace(calc=SimpleNamespace(results={"forces": forces}))
        for forces in [[[9, 0, 0]], [[0.04, 0, 0]], [[3, 4, 0]], [[0, 0, 0.3]]]
    ]
    assert check_mlneb_forces(history, frames) < 1e-12
    frames[2].calc.results["forces"] = [[3, 3, 0]]
    with pytest.raises(ValueError, match="does not match"):
        check_mlneb_forces(history, frames)


def test_conflicting_or_missing_steps_are_not_silently_pooled(tmp_path):
    path = tmp_path / "search.log"
    path.write_text(row(0, 1, 0.1) + row(0, 2, 0.2))
    with pytest.raises(ValueError, match="Conflicting"):
        read_mlneb_history(path)
    path.write_text(row(0, 1, 0.1) + row(2, 2, 0.2))
    with pytest.raises(ValueError, match="consecutive"):
        read_mlneb_history(path)
