The version file now lives at `chemparseplot/_version.py`, so the wheel no longer installs a top-level `_version.py`.
rgpycrumbs installs one there too, and uninstalling either package deleted the file of the other.
`chemparseplot.__version__` now reports the installed version.
