---
name: chemparseplot-docs
description: Find, edit and rebuild a page of the chemparseplot manual. Covers the Org pages under docs/orgmode/ that docs/export.el turns into generated reStructuredText, the MyST Markdown pages and notebooks under doc/source/, the API reference sphinx-autodoc2 writes from docstrings, links between pages, the site map doc/source/llms.txt and llms-full.txt, the README export, and the Emacs version the build needs. Use when asked to fix or add a docs page, when the file behind a page cannot be found, or when the docs build fails.
license: MIT
compatibility: scripts/where_is.py needs Python 3 and nothing else (checked with 3.12). Building the site needs Emacs 30 (Org 9.7), network access on the first export, and chemparseplot with its doc extra; running the tutorials needs the packages they import.
---

# The chemparseplot manual

## Which file is a page

| Pages | Source | Built by |
|---|---|---|
| `index`, `howto/`, `explanation/`, `reference/`, `dev/`, `release`, `used_by`, `tutorials/orca_neb` | `docs/orgmode/PAGE.org` | `docs/export.el` (ox-rst) writes `doc/source/PAGE.rst`, then Sphinx |
| `quickstart`, `installation`, `features`, the other `tutorials/`, `worklog/graphTrials` | `doc/source/PAGE.md` | Sphinx with MyST; pages with `{code-cell}` blocks are notebooks that myst-nb runs during the build |
| `apidocs/` | docstrings under `chemparseplot/` | sphinx-autodoc2, for every module, private ones included |
| `llms.txt` | `doc/source/llms.txt` | copied to the site root (`html_extra_path`) |

`python scripts/where_is.py PAGE` prints the source of one page, given
its address, its name (`howto/install`) or a file under `doc/source/`.

- Never edit `doc/source/**/*.rst`. They are written by the export and
  ignored by `doc/source/.gitignore`; the next export overwrites them.
- `tutorials/index` has both sources. `export.el` leaves
  `docs/orgmode/tutorials/index.org` out, so the site shows
  `doc/source/tutorials/index.md`.
- `doc/source/contributing.md` is left out by `exclude_patterns` in
  `doc/source/conf.py`. The published guide is
  `docs/orgmode/dev/contributing.org`; `CONTRIBUTING.md` at the root is
  empty.
- `readme.md` is exported from `readme_src.org` with
  `bash scripts/org_to_md.sh readme_src.org readme.md`. Change both. The
  Generate README workflow pushes its export to the `readme` branch,
  not to main.

## Navigation and links

- The toctrees are in the `#+begin_export rst` block at the end of
  `docs/orgmode/index.org`, one per section.
  `doc/source/tutorials/index.md` and
  `doc/source/tutorials/chemgp/index.md` hold MyST toctrees for their
  pages. The section pages such as `docs/orgmode/howto/index.org` are
  tables of links, not toctrees. A page in no toctree is built with the
  warning "document isn't included in any toctree", as
  `howto/surrogate_search_figures` is on main.
- An Org link to another page, `[[file:install.org][text]]`, comes
  out of ox-rst as `` `text <install.rst>`_ `` and in the HTML as
  `href="install.rst"`, which the site does not serve. Main has 111 such
  links in 25 Org files. Link pages with an export snippet instead:
  ``@@rst::doc:`install`@@`` (the target's title) or
  ``@@rst::doc:`text <install>`@@``, both of which become links to
  `install.html`.
- `python scripts/llms/build_llms.py` prints a note for each Org link
  to a file that does not exist; on main, `docs/orgmode/dev/index.org`
  links eight and `docs/orgmode/dev/contributing.org` one.

## Build

As CI does it (`.github/workflows/build_docs.yml`):

```bash
uv sync --extra doc
(cd docs && emacs --batch --load export.el)
uv run sphinx-build doc/source doc/build/html
```

- Emacs 30. CI installs 30.1. Emacs 29.3 with its Org 9.6.15 stops at
  the first page with `void-function (org-element-parent)`, although
  `export.el` defines a stand-in for `org-element-type-p`; and an
  ox-rst that Emacs 30 compiled stops under Emacs 29 with
  `wrong-number-of-arguments` in `org-element-type`. On the first run
  `export.el` installs ox-rst from MELPA into `~/.emacs.d/elpa`.
- Notebooks run during the build. A cell that fails is a warning, the
  build still succeeds, and the traceback is written under
  `doc/build/html/reports/`, which CI deploys with the rest of
  `doc/build/html`. CI installs the doc extra only, so all five
  notebooks stop at their first cell on a missing module (h5py,
  rgpycrumbs, ase or matplotlib). With `uv sync --all-extras` on main
  three fail: `tutorials/chemgp/convergence` and
  `tutorials/chemgp/surfaces` import pandas and plotnine, which no
  extra installs, and `tutorials/orca/geomscan` uses `TwoDimPlot`
  without importing it. With pandas and plotnine added,
  `tutorials/chemgp/surfaces` still fails: it passes `true_energy=` to
  `plot_gp_progression`, whose parameter is `_true_energy`.
- `uv run` and `uv sync` may rewrite `uv.lock`; restore it with
  `git checkout -- uv.lock` unless the dependencies changed.

## Site map and llms-full.txt

- Add a line for a new page to `doc/source/llms.txt`, in its section:
  `- [Title](https://chemparseplot.rgoswami.me/PAGE): what it is for`.
- Fetched without a browser, the pages of the site answer HTTP 403
  with a Cloudflare challenge ("Just a moment..."); only `llms.txt`
  comes back. Check a page in your own build,
  `doc/build/html/PAGE.html`, not on the site.
- `python scripts/llms/build_llms.py` writes `llms-full.txt`: every page
  in toctree order, converted to Markdown from its Org or MyST source,
  then `AGENTS.md` and the skills. Its settings are in
  `scripts/llms/llms.toml`. With `--check` it changes nothing and exits
  1 when the file is stale; CI runs that. It exits 2 when the site map
  links a page with no source or when a page names a file in a home,
  temporary or scratch directory.

## Check

```bash
python scripts/where_is.py
```

prints the source of every page the site map links and exits 1 when
one has none. On main all 39 have one.
