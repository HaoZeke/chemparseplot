`skills/` holds three task guides, each a `SKILL.md` with scripts that check what it says against the code.
`chemparseplot-orca-scan` reads an ORCA relaxed surface scan, shows that the distances `extract_energy_data` labels bohr are in ångström, and plots the scan; `chemparseplot-add-parser` covers adding a parser with its tests, fragment and docs pages; `chemparseplot-docs` finds the source of a page of the manual and covers the docs build.
`python scripts/skills/run_skills.py` checks every guide and runs its examples.
