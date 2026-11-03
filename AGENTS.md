# AGENTS.md

Working notes for changing or scripting against chemparseplot. The
manual is published at <https://chemparseplot.rgoswami.me/>, built from
Org pages under `docs/orgmode/` and MyST Markdown pages under
`doc/source/`. `doc/source/llms.txt` is its hand-written map, served as
<https://chemparseplot.rgoswami.me/llms.txt>, and `llms-full.txt` at the
repository root holds the manual in one file and names what it leaves
out. Task guides with scripts that CI runs are under `skills/`.

## What chemparseplot is

- Parsers that turn the output of computational chemistry codes into
  numpy arrays, pint quantities and small dataclasses: eOn (NEB,
  saddle searches, minimisations, `.con` frames through readcon), ORCA
  (relaxed surface scans, NEB through orca-pi or `.interp` files),
  Sella, ChemGP (HDF5 and JSONL), PLUMED, and surrogate-assisted
  searches (gpr_optim, ML-NEB).
- Plotting functions under `chemparseplot.plot`, on matplotlib, and on
  plotnine for `chemparseplot.plot.chemgp`.
- `chemparseplot.api` collects the entry points meant to stay stable:
  `parse_xyz`, `parse_orca_final_energy`,
  `extract_orca_geomscan_energy`, `load_trajectory`,
  `plot_con_overview`, `plot_con_overlay`, `suite_pins`.
- The command-line tools that draw these figures (`rgpycrumbs eon
  plt-neb` and the rest) and the surface fits live in rgpycrumbs, which
  chemparseplot imports inside a few plotting functions.

## Layout

| Path | Contents |
|---|---|
| `chemparseplot/parse/` | Parsers; a directory per code that has several (`eon`, `orca`, `sella`, `surrogate`, `trajectory`, `grammar`) |
| `chemparseplot/plot/` | Plotting; `structs.py` holds `EnergyPath`, `XYData`, `BasePlotter`, `TwoDimPlot` |
| `chemparseplot/units.py` | The pint registry: `Q_`, `ureg` and the energy-unit helpers |
| `tests/` | The pytest suite; small outputs in `tests/fixtures/` |
| `docs/orgmode/`, `docs/export.el` | Org pages and their export to reStructuredText |
| `doc/source/` | Sphinx: `conf.py`, MyST pages and notebooks, `llms.txt`; the exported `.rst` files and `apidocs/` are generated and ignored |
| `readme_src.org`, `readme.md` | The README and its Markdown export, which is also the PyPI description |
| `scripts/llms/` | The `llms-full.txt` generator, its settings `llms.toml` and its tests |
| `scripts/skills/run_skills.py` | Checks every skill and runs its examples |
| `skills/` | Task guides: `chemparseplot-orca-scan`, `chemparseplot-add-parser`, `chemparseplot-docs` |
| `doc/release/upcoming_changes/` | towncrier fragments; `CHANGELOG.md` is built from them |

## Install

- `pip install chemparseplot` installs numpy and pint and nothing
  else, which is enough for the parsers: in a bare install
  `chemparseplot.parse`, `chemparseplot.parse.orca` and
  `chemparseplot.api` import without loading any optional package.
  `chemparseplot.plot` needs matplotlib (the `plot` extra).
- The extras are `plot`, `neb`, `xyzrender`, `rings`, `opi`, `grammar`,
  `all` (those six), `doc`, `test`, `lint` and `release`. No extra
  installs rgpycrumbs except `test`, and none installs pandas or
  plotnine.
- For development, as CI does: `uv sync --all-extras` (Python 3.11 or
  3.12). `uv.lock` is out of date on main (`uv lock --check` fails), so
  `uv sync` and `uv run` rewrite it; restore it with
  `git checkout -- uv.lock` unless the change is about dependencies.
  The lock pins rgpycrumbs 1.3.0; PyPI has 1.13.0.
- The rewritten lock leaves the tree dirty, so hatch-vcs gives the
  package a dev version. At the v1.13.2 tag `uv sync --extra doc`
  installs `1.13.3.dev0+gd5e1b77cd.d` and the date instead of 1.13.2,
  and the title of the published site reads
  `chemparseplot 1.13.3.dev0+gd5e1b77cd.d20261031 documentation`.

## Test

- `uv run pytest tests`; CI runs `uv run pytest --cov=chemparseplot
  tests` on Python 3.11 and 3.12. At `d5e1b77` with all extras on
  Python 3.12: 528 passed and 59 skipped in 67 s. 58 of the skips are
  for pandas or plotnine.
- Run pytest through `uv run`, or with `.venv/bin` on `PATH`.
  `test_xyzrender_not_installed` takes a missing `xyzrender` command to
  mean that xyzrender is not installed. With the extra installed,
  `.venv/bin/python -m pytest` alone makes it fail: 527 passed, 59
  skipped and 1 failed.
- The markers select little. Of 585 tests, 25 are marked `pure`, 77
  `neb` and none `grammar`; 483 have no marker. `pytest -m pure` (22
  passed, 5 skipped, 2 s) is a smoke test, not the suite.
- The pandas and plotnine tests run in the pixi environments of
  `pixi.toml` (`test`, `plot`, `chemgp`, `plumed`), which the Coverage
  workflow combines; `pixi run cov` does the same locally.
- The Doctest Tutorials workflow runs `scripts/run_org_doctests.py` on
  `docs/orgmode/tutorials/*.org`. Every Python block there is marked
  `SKIP-DOCTEST`, so it runs none.

## Lint

- `uvx prek run -a`, as the lints workflow does: whitespace and
  end-of-file fixes, YAML and TOML checks, ruff (with `--fix`),
  ruff-format, taplo-format and codespell, configured in `prek.toml`.
  `uvx prek run --files FILE...` checks just the files you changed.
- The ruff rules are in `pyproject.toml`; `skills/ruff.toml` extends
  them with the allowances `scripts/` and `tests/` already have.

## Docs

```bash
uv sync --extra doc
(cd docs && emacs --batch --load export.el)
uv run sphinx-build doc/source doc/build/html
```

- The export needs Emacs 30 (Org 9.7); CI installs 30.1. Emacs 29.3,
  the Ubuntu 24.04 package, stops at the first page (see the
  `chemparseplot-docs` skill), and so does Emacs 30 with an ox-rst
  that Emacs 29 compiled. After switching Emacs, remove
  `~/.emacs.d/elpa/ox-rst-*` or use a fresh `HOME`. With Emacs 30.2 the
  export wrote 27 pages in 1.3 s, and in 4 to 6 s when it first
  installed ox-rst from MELPA; Sphinx, with the doc extra only, took
  21 s: "build succeeded, 17 warnings".
- Sphinx runs five notebooks: the two under `tutorials/chemgp/`,
  `tutorials/eon/saddle`, `tutorials/orca/geomscan` and
  `worklog/graphTrials`. A failing cell is a warning, not an error, and
  its traceback goes under `doc/build/html/reports/`. With the doc
  extra only, as CI installs it, all five stop at their first cell on
  a missing module (h5py, rgpycrumbs, ase or matplotlib). With
  `uv sync --all-extras` three still fail: the chemgp pages need pandas
  and plotnine, which no extra installs, and `tutorials/orca/geomscan`
  never imports `TwoDimPlot`.
- `python skills/chemparseplot-docs/scripts/where_is.py PAGE` names the
  file behind a page.
- Fetched without a browser in November 2026, the site served
  `llms.txt` but none of the 39 pages it links: each answered HTTP 403
  with a Cloudflare challenge titled "Just a moment...". An address the
  site does not have answers HTTP 200 with the home page, never a 404,
  so a fetched page is only real if its title matches. Read the pages
  from the repository, or `llms-full.txt`, which the site does not
  publish.
- `python scripts/llms/build_llms.py` rewrites `llms-full.txt` after a
  change to the pages, `AGENTS.md` or `skills/`; `--check`, which CI
  runs, exits 1 while it is stale. Its tests:
  `python -m unittest discover -s scripts/llms -p "test_*.py"`.
- `readme.md` is the export of `readme_src.org`
  (`bash scripts/org_to_md.sh readme_src.org readme.md`), but on main
  the two have drifted apart; change both by hand. Links in the README
  must be absolute, since PyPI shows it too.

## Changelog, commits, versions

- One towncrier fragment per change, in
  `doc/release/upcoming_changes/`: `ISSUE.TYPE.md`, or `+NAME.TYPE.md`
  without an issue, with `TYPE` one of `removed`, `deprecated`,
  `added`, `changed`, `fixed`, `misc`. Preview with
  `uv run towncrier build --draft --version X.Y.Z`. A release
  (`docs/orgmode/release.org`) moves the fragments into `CHANGELOG.md`
  below the `<!-- towncrier release notes start -->` line, so after
  v1.13.2 the directory does not exist until the next fragment. The
  file opens with 1.9.17 to 1.9.13, above that line; the newest entries
  start below it.
- Commit subjects are Conventional Commits with cog's default types
  (`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`,
  `ci`, `chore`, `revert`). No workflow checks them, but the release
  picks the next version with `cog bump --auto`, which reads the types
  since the latest tag (`cog.toml`). Release commits are
  `release: vX.Y.Z`.
- The version comes from the git tag through hatch-vcs and is written to
  the ignored `_version.py`. The `version` in `pixi.toml` and
  `current` in `tbump.toml` still say 1.8.0; the build does not read
  them. `doc/source/conf.py` asks the installed package and falls back
  to 1.11.0, so a docs build without chemparseplot installed shows that.
  A `v*` tag publishes to PyPI.

## Common tasks

- Read an ORCA scan, check its length unit and plot it: the
  `chemparseplot-orca-scan` skill.
- Add a parser, with its tests, fragment and docs: the
  `chemparseplot-add-parser` skill.
- Fix or add a docs page: the `chemparseplot-docs` skill.
- Check the skills: `uv run python scripts/skills/run_skills.py`
  (`--no-run` checks the frontmatter and links only).

## Pitfalls

- `geomscan.extract_energy_data` labels ORCA scan distances bohr, but
  ORCA printed them in ångström; `.to("angstrom")` then shrinks them
  1.89 times. The ORCA tutorial does that. Relabel instead.
- `extract_energy_data(text, energy_type)` matches the table name as
  written (`"Actual"`, `"SCF"`); `"actual"` returns two empty arrays and
  no error.
- `Q_(e, "hartree").to("kcal/mol")` raises `DimensionalityError`:
  pint's kcal/mol is per mole. Use `"kcal_mol"`, which chemparseplot
  defines, or `chemparseplot.units.convert_energy_magnitude`.
- `chemparseplot.plot.geomscan.plot_energy_paths` imports
  `rgpycrumbs.interpolation`, which needs scipy, and wants increasing
  distances; otherwise `ValueError: Error on input data`.
- `TwoDimPlot.add_data` takes `XYData`, not `EnergyPath`.
- `BasePlotter` calls `plt.style.use("bmh")` for the whole session.
- `import chemparseplot.plot.chemgp` fails without pandas and plotnine.
- `chemgp.plot_gp_progression` takes the true surface as its second
  positional argument, `_true_energy`. Its docstring and
  `tutorials/chemgp/surfaces` call it `true_energy`, and
  `true_energy=...` raises `TypeError`.
- Some docs say what the code does not:
  - `doc/source/installation.md` and `doc/source/quickstart.md` say
    that installing chemparseplot, or its `all` extra, pulls in
    rgpycrumbs, and that there is no `xyzrender` extra. Neither is
    true.
  - The quickstart calls `parse_eon_saddle(eresp="results_saddle.dat",
    rloc="saddle/")`, which raises `TypeError`. It takes the results
    directory as a `Path` and an `rgpycrumbs.basetypes.SpinID`, and the
    `SaddleMeasure` it returns has `saddle_energy` and
    `termination_status`, not `energy` and `status`.
  - `ARCHITECTURE.md` calls `extract_energy_data` with one argument and
    converts hartree to kcal/mol with `.to`; both raise.
  - `docs/orgmode/dev/contributing.org` puts tests next to the parser,
    lists a `plot` marker that is not declared, and imports
    `rgpycrumbs._aux.ensure_import`; rgpycrumbs is not a runtime
    dependency.
  - `howto/install`, `howto/faq` and `tutorials/orca_neb` list Python
    3.10; `requires-python` is `>=3.11`.
  - `howto/dev_testing` says `pixi run cov` fails below 90% coverage
    and that coverage runs with `parallel = true`. The `cov` task sets
    no threshold, the Coverage workflow fails below 70%, and
    `pyproject.toml` has `parallel = false`.
- An Org link to another page, `[[file:page.org][...]]`, ends up as a
  link to `page.rst` on the site; see the `chemparseplot-docs` skill.
- `CONTRIBUTING.md` at the root is empty; the guide is
  `docs/orgmode/dev/contributing.org`.
