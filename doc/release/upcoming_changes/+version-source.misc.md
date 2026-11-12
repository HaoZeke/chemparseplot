CI fails when `pixi.toml` sets a version field.
CI also fails when the newest `CHANGELOG.md` heading differs from the git tag that hatch-vcs reads.
A heading that does not parse, such as one with an empty version, fails as well.
Each linked heading must point at its own `vX.Y.Z` tag, and that tag must exist.
The new `scripts/check_version.py` script runs these checks.
The stale `tbump.toml` no longer exists, and the release headings link to the `v`-prefixed tags.
The 1.4.0 heading has no link, since no `v1.4.0` tag exists.
A `release: vX.Y.Z` commit passes only when that version is above the latest tag.
The release recipe takes that version from `cog bump --dry-run --auto` instead of a fixed 1.7.1.
The recipe stops before towncrier when cog fails or prints anything other than `vX.Y.Z`.
It passes `--frozen` to `uv sync` and `uv run`, so `uv.lock` stays as committed.
