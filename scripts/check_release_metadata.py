"""Validate compatibility requirements in built chemparseplot wheels."""

from __future__ import annotations

import argparse
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

_READCON_REQUIREMENT = re.compile(
    r"^readcon>=0\.14\.7(?:[^;]*)?;\s*extra\s*==\s*['\"](?P<extra>neb|test)['\"]$",
    re.IGNORECASE,
)


def check_wheel_metadata(wheel: Path) -> None:
    """Require spec-three readcon metadata for the eOn-facing extras."""
    with ZipFile(wheel) as archive:
        metadata_paths = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            message = f"expected one wheel METADATA file in {wheel}"
            raise ValueError(message)
        message = BytesParser(policy=policy.compat32).parsebytes(
            archive.read(metadata_paths[0])
        )

    found = {
        match.group("extra").lower()
        for requirement in message.get_all("Requires-Dist", [])
        if (match := _READCON_REQUIREMENT.match(requirement))
    }
    missing = {"neb", "test"} - found
    if missing:
        names = ", ".join(sorted(missing))
        error = f"{wheel} lacks readcon>=0.14.7 for extra(s): {names}"
        raise ValueError(error)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    for wheel in args.wheels:
        check_wheel_metadata(wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
