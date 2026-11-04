---
name: chemparseplot-add-parser
description: Add a parser for the output of a computational chemistry code to chemparseplot. Covers where the module and its tests go, keeping the bare install (numpy and pint) importable, returning pint quantities in the unit the code printed, failing loudly instead of returning empty arrays, the MyST docstring behind the API reference, the test markers, the towncrier fragment and the docs pages to touch. Use when asked to parse a new type of output file, or a new section of ORCA, eOn, Sella or PLUMED output.
license: See LICENSE in the chemparseplot repository
compatibility: A chemparseplot checkout with Python 3.11 or newer and uv (`uv sync --all-extras`). The check script needs pytest, ruff and towncrier from that environment and runs offline.
---

# Adding a chemparseplot parser

## Where it goes

- The module is `chemparseplot/parse/ENGINE/NAME.py` for a code that already has a directory (`eon`, `orca`, `sella`, `surrogate`, `trajectory`), or `chemparseplot/parse/NAME.py` next to `plumed.py` and `chemgp_hdf5.py`.
- The tests are `tests/parse/ENGINE/test_NAME.py`, or `tests/parse/test_NAME.py`.
  pytest collects `tests/` only.
  The layout in `docs/orgmode/dev/contributing.org`, with the test next to the parser, is not what the repository does.
- Re-exporting from the package `__init__.py` is optional.
  `chemparseplot.parse.orca` exports `geomscan` and the nudged elastic band (NEB) parsers, but callers import most modules by their full name.

## The module

[assets/scan_steps.py](assets/scan_steps.py) is a complete parser.
It reads the banner that the quantum chemistry program (ORCA) prints at each step of a bond scan and returns the atom pair and the bond lengths:

- The Software Package Data Exchange (SPDX) header and a module docstring, like every file in the package.
- Only numpy, pint and the standard library at module level.
  The dependencies of chemparseplot are numpy and pint.
  `import chemparseplot.parse` loads `eon`, `orca` and `patterns`, and `chemparseplot.parse.orca` loads `geomscan` and `neb`.
  A module-level import of anything else in a module they load breaks every bare install.
  Import optional packages inside the function that needs them (ruff's PLC0415 is switched off under `chemparseplot/parse/`).
  rgpycrumbs is not a dependency either.
  The `rgpycrumbs._aux.ensure_import` recipe in `docs/orgmode/dev/contributing.org` fails where rgpycrumbs is not installed.
- Values as pint quantities from `chemparseplot.units.Q_`, in the unit the code printed.
  Check it against the output instead of the code's defaults: `geomscan.extract_energy_data` labels ORCA scan distances bohr while ORCA printed ångström (see the `chemparseplot-orca-scan` skill).
  Unit names are pint's: `"hartree"`, `"eV"`, `"angstrom"`, `"bohr"`.
  `"kcal/mol"` is per mole in pint and does not convert from hartree.
  `"kcal_mol"`, which chemparseplot defines, does.
- A `ValueError` with a message when the text has nothing to parse.
  `geomscan.extract_energy_data` returns empty arrays instead, and a misspelt table name then looks like an empty scan.
  Build messages in a variable first (`msg = ...; raise ValueError(msg)`), since ruff's EM rules are on.
- A numpy-style docstring with a MyST block for the first release with the function:

  ````text
  ```{versionadded} 1.14.0
  ```
  ````

  The version comes from git tags (hatch-vcs).
  Take the release after `git describe --tags --abbrev=0`.
  sphinx-autodoc2 writes the API reference at `/apidocs/` from every module of the package, so the docstring is the reference page and there is nothing to register.

## Tests

[assets/test_scan_steps.py](assets/test_scan_steps.py) shows the usual shape: a snippet of real output inline, the error paths, and a check against a whole run when one is in the repository.

- Mark the module `pytestmark = pytest.mark.pure` when the parser needs only numpy and pint.
  pyproject.toml declares three markers: `pure`, `neb` and `grammar`.
  Tests marked `neb` need ase, polars, h5py, scipy, matplotlib and rgpycrumbs.
  Call `skip_if_not_env("neb")` from `tests.conftest` at the top of such a module, as `tests/parse/test_neb_utils.py` does.
  Tests marked `grammar` need parsimonious.
  The `plot` marker that `docs/orgmode/dev/contributing.org` lists is not declared.
- Keep snippets inline or in `tests/fixtures/`.
  A whole output goes in gzipped, like `doc/source/tutorials/orca/data/scanuhf.gz`.
  The pre-commit hooks refuse files over 1000 kB.
- Run one file with `uv run pytest tests/parse/orca/test_scan_steps.py`.
  CI runs `uv sync --all-extras` and `uv run pytest --cov=chemparseplot tests` on Python 3.11 and 3.12.
  `uv run` and `uv sync` may rewrite `uv.lock`.
  Leave that out of the commit unless the dependencies changed.

## Lint, changelog and docs

- `uv run ruff check` and `uv run ruff format` on the new files, or `uvx prek run --files FILE...` for all the hooks of `prek.toml` (ruff, ruff-format, taplo, codespell, whitespace).
- A towncrier fragment in `doc/release/upcoming_changes/`.
  A release moves the fragments into `CHANGELOG.md`, so on main the directory may not exist until the first fragment is added: `ISSUE.added.md`, or `+NAME.added.md` without an issue.
  One line or paragraph of Markdown.
  The types are `removed`, `deprecated`, `added`, `changed`, `fixed` and `misc`.
  Preview with `uv run towncrier build --draft --version X.Y.Z`.
  The draft changes nothing.
- A how-to page when the parser is meant for users: `docs/orgmode/howto/NAME.org`, added to the How-to toctree in the `#+begin_export rst` block of `docs/orgmode/index.org` and to the table in `docs/orgmode/howto/index.org`, plus a line in the site map `doc/source/llms.txt`.
  Then run `python scripts/llms/build_llms.py`.
  CI fails while `llms-full.txt` is stale.
- Commit subjects follow the log: `feat(parse): read ORCA scan step banners`.

## Check

```bash
python scripts/scaffold_check.py
```

The script copies the package, the tests, `pyproject.toml`, `CHANGELOG.md` and the ORCA tutorial run into a temporary directory.
It puts the two assets at `chemparseplot/parse/orca/scan_steps.py` and `tests/parse/orca/test_scan_steps.py`, and adds a fragment for them.
It then checks that Python imports the module from the copy and that its four tests pass.
It also checks that ruff check and ruff format accept both files under the repository's rules.
Last, a towncrier draft has to list the fragment under "Added".
The checkout is not changed.
