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

Your contributions make this project better for everyone. Thank you for
participating!

### Local Development

Often it is useful to have [pixi](https://prefix.dev/) handle the dependencies:

```bash
pixi shell
```

A `pre-commit` job is set up on CI to enforce consistent styles. It is advisable
to set it up locally as well using [pipx](https://pypa.github.io/pipx/) for
isolation:

```bash
# Run before committing
pipx run pre-commit run --all-files
# Or install the git hook to enforce this
pipx run pre-commit install
```

Note that the `readme` file is generated from `readme_src.org` via:

```bash
./scripts/org_to_md.sh readme_src.org readme.md
```

#### Tests

Tests and checks are run on the CI, however locally one can use:

```bash
uv run pytest --cov=chemparseplot tests
```


#### Documentation

We use `sphinx` with the `myst-parser`:

```bash
sphinx-build doc/source doc/build/html
python -m http.server -d doc/build/html
```
