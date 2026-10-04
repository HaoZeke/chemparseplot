"""Figure files are byte-identical for equal inputs and name their inputs."""

import json

import matplotlib as mpl
import pytest

mpl.use("Agg")
import matplotlib.pyplot as plt

from chemparseplot.plot.provenance import file_sha256, save_with_provenance


def _fig():
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([1, 2, 3], [3, 1, 2])
    ax.set_title("t")
    return fig


@pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
def test_equal_inputs_give_equal_bytes(tmp_path, fmt):
    src = tmp_path / "in.txt"
    src.write_text("data")
    a, b = tmp_path / f"a.{fmt}", tmp_path / f"b.{fmt}"
    save_with_provenance(_fig(), a, {"in": src}, command="x")
    save_with_provenance(_fig(), b, {"in": src}, command="x")
    plt.close("all")
    assert file_sha256(a) == file_sha256(b)


def test_record_names_inputs_and_versions(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("data")
    out = tmp_path / "f.png"
    rec = save_with_provenance(_fig(), out, {"in": src}, command="plt x")
    plt.close("all")
    assert rec["inputs"]["in"] == file_sha256(src)
    assert "matplotlib" in rec["tools"] and rec["command"] == "plt x"
    side = json.loads((tmp_path / "f.png.provenance.json").read_text())
    assert side["output"]["sha256"] == file_sha256(out)
    assert b"matplotlib" in out.read_bytes()  # the description is embedded


def test_unknown_suffix_refused(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        save_with_provenance(_fig(), tmp_path / "f.gif", {})
    plt.close("all")


def test_precomputed_hashes_are_merged(tmp_path):
    rec = save_with_provenance(
        _fig(), tmp_path / "f.svg", {}, hashes={"band.h5": "ab" * 32}, sidecar=False
    )
    plt.close("all")
    assert rec["inputs"] == {"band.h5": "ab" * 32}
    assert not (tmp_path / "f.svg.provenance.json").exists()
