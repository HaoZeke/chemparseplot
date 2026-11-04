"""Add the example parser to a scratch copy of chemparseplot and check it.

    python scripts/scaffold_check.py

Copies the package, its tests, pyproject.toml, CHANGELOG.md and the ORCA
tutorial run into a temporary directory, puts assets/scan_steps.py at
chemparseplot/parse/orca/scan_steps.py and assets/test_scan_steps.py at
tests/parse/orca/test_scan_steps.py, and checks in that copy that

- Python imports the new module from the copy;
- its four tests pass under the repository's pytest settings;
- ruff check and ruff format --check pass under the repository's rules;
- a changelog fragment appears under "Added" in a towncrier draft.

The checkout is not changed. Run it with the interpreter of
`uv sync --all-extras`, which has pytest, ruff and towncrier.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
ROOT = SKILL.parents[1]
MODULE = Path("chemparseplot/parse/orca/scan_steps.py")
TEST = Path("tests/parse/orca/test_scan_steps.py")
TUTORIAL_RUN = Path("doc/source/tutorials/orca/data/scanuhf.gz")
FRAGMENT = Path("doc/release/upcoming_changes/+scan_steps.added.md")
FRAGMENT_TEXT = (
    "`chemparseplot.parse.orca.scan_steps.extract_scan_steps` reads the atom pair"
    " and the bond length of every step of an ORCA bond scan."
)
TESTS_EXPECTED = "4 passed"


def scratch_copy(dest: Path) -> None:
    skip = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "chemparseplot", dest / "chemparseplot", ignore=skip)
    shutil.copytree(ROOT / "tests", dest / "tests", ignore=skip)
    for name in ("pyproject.toml", "CHANGELOG.md"):
        shutil.copy2(ROOT / name, dest / name)
    (dest / TUTORIAL_RUN).parent.mkdir(parents=True)
    shutil.copy2(ROOT / TUTORIAL_RUN, dest / TUTORIAL_RUN)
    shutil.copy2(SKILL / "assets/scan_steps.py", dest / MODULE)
    shutil.copy2(SKILL / "assets/test_scan_steps.py", dest / TEST)
    (dest / FRAGMENT).parent.mkdir(parents=True)
    (dest / FRAGMENT).write_text(FRAGMENT_TEXT + "\n", encoding="utf-8")


def run(label: str, cmd: list[str], cwd: Path, env: dict[str, str]) -> str | None:
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(f"FAILED: {label} (exit status {proc.returncode})\n{out}")
        return None
    return out


def main() -> int:
    if 'name = "chemparseplot"' not in (ROOT / "pyproject.toml").read_text():
        sys.exit(f"{ROOT} is not a chemparseplot checkout")
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp)
        scratch_copy(copy)
        env = dict(os.environ, PYTHONPATH=str(copy), PYTHONDONTWRITEBYTECODE="1")
        py = sys.executable
        failures = 0

        out = run(
            "import the new module",
            [
                py,
                "-c",
                "import chemparseplot.parse.orca.scan_steps as m; print(m.__file__)",
            ],
            copy,
            env,
        )
        if out is not None and Path(out.strip()).resolve() == (copy / MODULE).resolve():
            print(f"import: {MODULE} comes from the scratch copy")
        else:
            failures += 1
            if out is not None:
                print(f"FAILED: the module was imported from {out.strip()}")

        out = run(
            "pytest",
            [py, "-m", "pytest", "-p", "no:cacheprovider", str(TEST)],
            copy,
            env,
        )
        summary = out.strip().splitlines()[-1] if out else ""
        if out is not None and summary.startswith(TESTS_EXPECTED):
            print(f"pytest {TEST}: {summary}")
        else:
            failures += 1
            if out is not None:
                print(f"FAILED: pytest ended with {summary!r}, not {TESTS_EXPECTED!r}")

        for label, args in (
            ("ruff check", ["check"]),
            ("ruff format", ["format", "--check"]),
        ):
            out = run(label, [py, "-m", "ruff", *args, str(MODULE), str(TEST)], copy, env)
            if out is None:
                failures += 1
            else:
                print(f"{label}: {out.strip().splitlines()[-1]}")

        out = run(
            "towncrier draft",
            [py, "-m", "towncrier", "build", "--draft", "--version", "0.0.0"],
            copy,
            env,
        )
        if out is not None and "### Added" in out and FRAGMENT_TEXT in out:
            print(f"towncrier: {FRAGMENT.name} is listed under Added")
        else:
            failures += 1
            if out is not None:
                print(f"FAILED: the draft does not list the fragment\n{out}")

    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
