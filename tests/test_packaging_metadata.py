import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.check_release_metadata import check_wheel_metadata


def test_readcon_spec_three_floor_is_declared_for_reader_extras():
    root = Path(__file__).parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]

    assert "readcon>=0.14.7" in extras["neb"]
    assert "readcon>=0.14.7" in extras["test"]


def test_built_wheel_preserves_readcon_floor(tmp_path):
    wheel = tmp_path / "chemparseplot-1.12.0-py3-none-any.whl"
    metadata = """Metadata-Version: 2.3
Name: chemparseplot
Version: 1.12.0
Provides-Extra: neb
Requires-Dist: readcon>=0.14.7; extra == 'neb'
Provides-Extra: test
Requires-Dist: readcon>=0.14.7; extra == 'test'
"""
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            "chemparseplot-1.12.0.dist-info/METADATA", metadata
        )

    check_wheel_metadata(wheel)


def test_built_wheel_rejects_old_readcon_floor(tmp_path):
    wheel = tmp_path / "chemparseplot-1.12.0-py3-none-any.whl"
    metadata = """Metadata-Version: 2.3
Name: chemparseplot
Version: 1.12.0
Provides-Extra: neb
Requires-Dist: readcon>=0.13.1; extra == 'neb'
Provides-Extra: test
Requires-Dist: readcon>=0.13.1; extra == 'test'
"""
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            "chemparseplot-1.12.0.dist-info/METADATA", metadata
        )

    with pytest.raises(ValueError, match="readcon>=0.14.7"):
        check_wheel_metadata(wheel)
