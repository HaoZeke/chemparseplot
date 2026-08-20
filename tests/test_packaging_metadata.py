from pathlib import Path
import tomllib


def test_readcon_spec_three_floor_is_declared_for_reader_extras():
    root = Path(__file__).parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]

    assert "readcon>=0.14.7" in extras["neb"]
    assert "readcon>=0.14.7" in extras["test"]
