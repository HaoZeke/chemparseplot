# SPDX-License-Identifier: MIT
"""scripts/check_version.py against throwaway git histories."""

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_version.py"
MARKER = "<!-- towncrier release notes start -->"
RELEASE_DATE = "2026-01-01"
TITLE_FORMAT = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["towncrier"][
    "title_format"
]
GIT_ENV = {
    "GIT_AUTHOR_NAME": "check-version",
    "GIT_AUTHOR_EMAIL": "",
    "GIT_COMMITTER_NAME": "check-version",
    "GIT_COMMITTER_EMAIL": "",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def git(repo, *args):
    env = {**os.environ, **GIT_ENV}
    subprocess.run(
        ["git", *args],  # noqa: S607
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )


def title(version):
    return TITLE_FORMAT.format(version=version, project_date=RELEASE_DATE)


def add_heading(repo, heading):
    """Insert *heading* as the newest entry below the towncrier marker."""
    changelog = repo / "CHANGELOG.md"
    head, marker, notes = changelog.read_text().partition(MARKER)
    changelog.write_text(f"{head}{marker}\n\n{heading}\n\n- Notes.\n{notes}")


def append_heading(repo, heading):
    """Add *heading* as the oldest entry."""
    changelog = repo / "CHANGELOG.md"
    changelog.write_text(f"{changelog.read_text()}\n{heading}\n\n- Notes.\n")


def v_label(version):
    """Return the title with a ``v`` on its label, as the 1.x headings have."""
    return title(version).replace(f"[{version}]", f"[v{version}]")


def release(repo, version):
    add_heading(repo, title(version))
    git(repo, "commit", "-q", "-am", f"release: v{version}")


def check(repo, *args):
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_version.py"), *args],
        cwd=repo,
        env={**os.environ, **GIT_ENV},
        check=False,
        capture_output=True,
        text=True,
    )


def init_repo(path, version):
    """Return a repository at *path* whose only commit is ``release: v<version>``."""
    (path / "scripts").mkdir()
    shutil.copy(SCRIPT, path / "scripts" / "check_version.py")
    (path / "pixi.toml").write_text('[workspace]\nname = "chemparseplot"\n')
    (path / "pyproject.toml").write_text(
        f"[tool.towncrier]\ntitle_format = {TITLE_FORMAT!r}\n"
    )
    (path / "CHANGELOG.md").write_text(f"# Changelog\n\n{MARKER}\n")
    git(path, "init", "-q")
    git(path, "add", ".")
    release(path, version)
    return path


@pytest.fixture
def repo(tmp_path):
    init_repo(tmp_path, "1.0.0")
    git(tmp_path, "tag", "v1.0.0")
    return tmp_path


def test_tagged_release_passes(repo):
    result = check(repo)
    assert result.returncode == 0, result.stderr
    assert "version 1.0.0" in result.stdout


def test_release_commit_passes_before_its_tag(repo):
    release(repo, "1.1.0")
    result = check(repo)
    assert result.returncode == 0, result.stderr
    assert "version 1.1.0" in result.stdout


def test_empty_release_heading_fails(repo):
    """An empty cog version gives ``release: v`` and an empty heading."""
    release(repo, "")
    result = check(repo)
    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        f"version drift: CHANGELOG.md heading cannot be parsed: {title('')}"
    ]
    assert "all copies agree" not in result.stdout


def test_release_heading_with_the_wrong_number_fails(repo):
    add_heading(repo, title("1.2.0"))
    git(repo, "commit", "-q", "-am", "release: v1.1.0")
    result = check(repo)
    assert result.returncode == 1
    assert "heading is 1.2.0, expected 1.0.0" in result.stderr


def test_release_heading_with_the_right_number_passes(repo):
    release(repo, "1.0.1")
    result = check(repo)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "version 1.0.1: all copies agree\n"


def test_release_commit_below_the_tag_fails(repo):
    release(repo, "0.9.0")
    result = check(repo)
    assert result.returncode == 1
    assert "release: v0.9.0 is not newer than v1.0.0" in result.stderr


def test_release_commit_equal_to_the_tag_fails(repo):
    release(repo, "1.0.0")
    result = check(repo)
    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        "version drift: release: v1.0.0 is not newer than v1.0.0"
    ]


def test_release_commit_compares_numbers(repo):
    release(repo, "1.9.0")
    git(repo, "tag", "v1.9.0")
    release(repo, "1.10.0")
    result = check(repo)
    assert result.returncode == 0, result.stderr
    assert "version 1.10.0" in result.stdout


def test_deleted_tag_fails_after_the_release_commit(repo):
    release(repo, "1.1.0")
    git(repo, "tag", "v1.1.0")
    git(repo, "commit", "-q", "--allow-empty", "-m", "docs: after the release")
    assert check(repo).returncode == 0

    git(repo, "tag", "-d", "v1.1.0")
    result = check(repo)
    assert result.returncode == 1
    assert "heading is 1.1.0, expected 1.0.0" in result.stderr
    assert "heading 1.1.0 links to a missing tag" in result.stderr


def test_no_reachable_tag_fails(repo):
    git(repo, "tag", "-d", "v1.0.0")
    git(repo, "commit", "-q", "--allow-empty", "-m", "docs: untagged")
    result = check(repo)
    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        "version drift: no v* tag is reachable from HEAD"
    ]


def test_untagged_release_commit_fails(tmp_path):
    repo = init_repo(tmp_path, "1.2.3")
    result = check(repo)
    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        "version drift: no v* tag is reachable from HEAD"
    ]
    assert "all copies agree" not in result.stdout


def test_untagged_release_commit_passes_with_an_expected_tag(tmp_path):
    repo = init_repo(tmp_path, "1.2.3")
    result = check(repo, "v1.2.3")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "version 1.2.3: all copies agree\n"


def test_v_label_with_a_bare_ref_fails(repo):
    git(repo, "tag", "v0.9.0")
    append_heading(repo, v_label("0.9.0").replace("tree/v0.9.0", "tree/0.9.0"))
    result = check(repo)
    assert result.returncode == 1
    assert "heading 0.9.0 links to" in result.stderr


def test_v_label_with_its_tag_passes(repo):
    git(repo, "tag", "v0.9.0")
    append_heading(repo, v_label("0.9.0"))
    result = check(repo)
    assert result.returncode == 0, result.stderr


def test_linked_heading_without_a_tag_fails(repo):
    append_heading(repo, title("0.9.0"))
    result = check(repo)
    assert result.returncode == 1
    assert "heading 0.9.0 links to a missing tag" in result.stderr


def test_unlinked_heading_without_a_tag_passes(repo):
    append_heading(repo, f"## 0.9.0 - {RELEASE_DATE}")
    result = check(repo)
    assert result.returncode == 0, result.stderr


def test_pixi_version_fails(repo):
    (repo / "pixi.toml").write_text(
        '[workspace]\nname = "chemparseplot"\nversion = "1.0.0"\n'
    )
    result = check(repo)
    assert result.returncode == 1
    assert "pixi.toml sets workspace.version" in result.stderr


def test_explicit_tag_must_match_the_heading(repo):
    assert check(repo, "v1.0.0").returncode == 0
    result = check(repo, "v1.0.1")
    assert result.returncode == 1
    assert "heading is 1.0.0, expected 1.0.1" in result.stderr
