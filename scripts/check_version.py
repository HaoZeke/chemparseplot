#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail when a version copy disagrees with the git tag.

hatch-vcs derives the package version from the latest ``vX.Y.Z`` tag, so no
tracked file carries its own version field. The towncrier heading is the only
version typed at release time. It must match the latest tag reachable from
HEAD. The one exception is the release commit itself: when the subject of HEAD
is ``release: vX.Y.Z`` for the newest heading, that heading may run ahead of
the tag, because the tag will point at HEAD. The heading must then be strictly
greater than the latest tag; ``cog bump --dry-run --auto`` picks that version
in the release recipe, and the lints job has no cog to ask. An older
``release:`` commit never stands in for a missing tag.

Every ``## `` line below the towncrier marker must parse as a release heading.
An empty version, as in ``## [](.../tree/v)``, is an error, and the check never
falls through to an older heading. Every linked heading, with or without a
``v`` on its label, must link where the towncrier ``title_format`` would, and
its tag must exist. A heading without a link is not checked against a tag.

With no reachable ``v*`` tag the script reports only that, on a release commit
too, unless EXPECTED is given.

Usage: python scripts/check_version.py [EXPECTED]
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = "<!-- towncrier release notes start -->"
HEADING = re.compile(r"## (?:\[v?(\d+\.\d+\.\d+)\]\(([^)]*)\)|v?(\d+\.\d+\.\d+)\b)")
LINK = re.compile(r"\]\(([^)]*)\)")


def git(*args):
    return subprocess.run(
        ["git", *args],  # noqa: S607
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def latest_tag():
    try:
        return git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*")
    except subprocess.CalledProcessError:
        return None


def parse_headings(notes, errors):
    """Return ``(version, link)`` for each ``## `` line in *notes*, or None
    for a line that does not parse, which is also reported in *errors*."""
    headings = []
    for line in notes.splitlines():
        if not line.startswith("## "):
            continue
        match = HEADING.match(line)
        if match is None:
            errors.append(f"CHANGELOG.md heading cannot be parsed: {line}")
            headings.append(None)
            continue
        linked, link, unlinked = match.groups()
        headings.append((linked or unlinked, link))
    return headings


def release_number(version):
    return tuple(int(part) for part in version.removeprefix("v").split("."))


def pending_release(heading, tag, errors):
    """Return *heading* when HEAD is its release commit and *heading* is newer
    than *tag*, else None. A release commit at or below *tag* is an error
    unless *tag* already points at HEAD. With no *tag* it returns None, since
    hatch-vcs then has no tag to read."""
    if tag is None or not heading:
        return None
    if git("log", "-1", "--format=%s") != f"release: v{heading}":
        return None
    if release_number(heading) > release_number(tag):
        return heading
    if f"v{heading}" not in git("tag", "--points-at", "HEAD").split():
        errors.append(f"release: v{heading} is not newer than {tag}")
    return None


def expected_version(pending, tag):
    """Return the version HEAD builds as, or None when no tag is reachable."""
    if pending:
        return pending
    return tag.removeprefix("v") if tag else None


def main(argv):
    errors = []
    with (ROOT / "pixi.toml").open("rb") as f:
        workspace = tomllib.load(f).get("workspace", {})
    if "version" in workspace:
        errors.append(f"pixi.toml sets workspace.version = {workspace['version']!r}")
    with (ROOT / "pyproject.toml").open("rb") as f:
        title_format = tomllib.load(f)["tool"]["towncrier"]["title_format"]
    link_format = LINK.search(title_format).group(1)

    notes = (ROOT / "CHANGELOG.md").read_text().partition(MARKER)[2]
    headings = parse_headings(notes, errors)
    unparsed = bool(headings) and headings[0] is None
    heading = headings[0][0] if headings and not unparsed else None
    tag = latest_tag()
    pending = None if argv else pending_release(heading, tag, errors)
    expected = argv[0].removeprefix("v") if argv else expected_version(pending, tag)
    if expected is None:
        errors.append("no v* tag is reachable from HEAD")
    elif heading != expected and not unparsed:
        errors.append(f"CHANGELOG.md heading is {heading}, expected {expected}")

    tags = set(git("tag", "--list", "v[0-9]*").splitlines())
    for version, link in filter(None, headings):
        if not link:
            continue
        if link != link_format.format(version=version):
            errors.append(f"CHANGELOG.md heading {version} links to {link}")
        if tags and f"v{version}" not in tags and version != pending:
            errors.append(f"CHANGELOG.md heading {version} links to a missing tag")

    for error in errors:
        print(f"version drift: {error}", file=sys.stderr)
    if not errors:
        print(f"version {expected}: all copies agree")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
