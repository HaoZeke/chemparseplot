# Contributing to `chemparseplot`

Thank you for considering contributing to `chemparseplot`! We appreciate your
interest in making this project better.

## Code of Conduct

Please read and adhere to our [Code of Conduct](https://github.com/HaoZeke/chemparseplot/blob/main/CODE_OF_CONDUCT.md) to maintain
a safe, welcoming, and inclusive environment.

## Types of Contributions

We welcome various forms of contributions:

- **Bug Reports**: Feel free to report any bugs you encounter.
- **Documentation**: Typos, clarity issues, or missing guides—your help is
  welcome here.
- **Feature Discussions/Requests**: Got an idea? Open an issue to discuss its
  potential.
- **Code Contributions**: All code contributions are welcome.

## Using Co-Authored-By in Git Commits

We encourage the use of [co-authored
commits](https://docs.github.com/en/github/committing-changes-to-your-project/creating-a-commit-with-multiple-authors)
for collaborative efforts. This helps in giving credit to all contributors for
their work.

```markdown
Co-authored-by: name <name@example.com>
Co-authored-by: another-name <another-name@example.com>
```

## Development

This project uses [`uv`](https://docs.astral.sh/uv/) as the primary development
tool with [`hatchling`](https://hatch.pypa.io/) +
[`hatch-vcs`](https://github.com/ofek/hatch-vcs) for building and versioning.

### Setup

```bash
# Install in development mode with test and plotting dependencies
uv sync --extra test --extra plot

# Run tests
uv run pytest --cov=chemparseplot tests
```

The `rgpycrumbs` dependency is resolved from git via `[tool.uv.sources]` in
`pyproject.toml`, so `uv sync` automatically fetches the latest version from
the main branch.

### Pre-commit

A `pre-commit` job is set up on CI to enforce consistent styles. It is advisable
to set it up locally as well using [pipx](https://pypa.github.io/pipx/) for
isolation:

```bash
# Run before committing
pipx run pre-commit run --all-files
# Or install the git hook to enforce this
pipx run pre-commit install
```

### README

The `readme` file is generated from `readme_src.org` via:

```bash
./scripts/org_to_md.sh readme_src.org readme.md
```

### Versioning

Versions are derived automatically from **git tags** via `hatch-vcs`
(setuptools-scm). There is no manual version field; the version is the latest
tag (e.g. `v1.0.0` → `1.0.0`). Between tags, dev versions are generated
automatically (e.g. `1.0.1.dev3+gabcdef`).

### Release Process

```bash
# 1. Ensure tests pass
uv run pytest --cov=chemparseplot tests

# 2. Build changelog (uses towncrier fragments in doc/release/upcoming_changes/)
uvx towncrier build --version "v1.0.0"

# 3. Commit the changelog
git add CHANGELOG.md && git commit -m "doc: release notes for v1.0.0"

# 4. Tag the release (hatch-vcs derives the version from this tag)
git tag -a v1.0.0 -m "Version 1.0.0"

# 5. Build and publish
uvx hatch build
uvx hatch publish
```

### Documentation

We use `sphinx` with the `myst-parser`:

```bash
uv sync --extra doc
uv run sphinx-build doc/source doc/build/html
python -m http.server -d doc/build/html
```
