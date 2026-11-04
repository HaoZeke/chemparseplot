`llms-full.txt` at the repository root holds the manual in one Markdown file, the pages in the order of the site's navigation, converted from the Org and MyST sources by `python scripts/llms/build_llms.py` (settings in `scripts/llms/llms.toml`).
Its header names what it leaves out: the API reference, and any page listed under `full_exclude`.
`--check` exits 1 while the file is stale, and 2 when the site map `doc/source/llms.txt` links a page that does not exist.
