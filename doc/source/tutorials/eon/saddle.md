---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# eOn Saddle Search Analysis

This tutorial shows how to parse saddle search results from eOn calculations.

## Prerequisites

```{code-cell} ipython3
from pathlib import Path
from chemparseplot.parse.eon.saddle_search import parse_eon_saddle
from rgpycrumbs.basetypes import SpinID
```

## Parsing Saddle Search Results

eOn stores saddle search results in directories with `results.dat` and log files.

## Example Usage

```{code-cell} ipython3
# Point to your eOn results directory
# results_dir = Path("/path/to/eon/results/")

# Create a SpinID for the location
# rloc = SpinID(mol_id="H2O", spin=0)

# Parse the saddle search results
# saddle_data = parse_eon_saddle(results_dir, rloc)

# Access results
# print(f"Success: {saddle_data.success}")
# print(f"Barrier: {saddle_data.barrier}")
# print(f"Method: {saddle_data.method}")
```

## Accessing Saddle Search Data

```{code-cell} ipython3
# SaddleMeasure provides:
# - success: bool - whether the search converged
# - barrier: Q_ - energy barrier with units
# - saddle_energy: Q_ - saddle point energy
# - init_energy: Q_ - initial state energy
# - pes_calls: int - number of PES evaluations
# - iter_steps: int - optimization iterations
# - tot_time: float - wall time in seconds
# - method: DimerOpt - optimization method details
# - termination_status: str - reason for termination
```

## Batch Processing

```{code-cell} ipython3
# Process multiple saddle searches
# from glob import glob
#
# results_dirs = glob("saddle_searches/*/")
# all_results = []
#
# for i, res_dir in enumerate(results_dirs):
#     rloc = SpinID(mol_id=f"mol_{i}", spin=0)
#     result = parse_eon_saddle(Path(res_dir), rloc)
#     all_results.append(result)
#
# print(f"Processed {len(all_results)} saddle searches")
# print(f"Success rate: {sum(r.success for r in all_results) / len(all_results):.1%}")
```
