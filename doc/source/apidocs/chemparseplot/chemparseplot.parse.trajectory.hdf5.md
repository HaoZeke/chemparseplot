# {py:mod}`chemparseplot.parse.trajectory.hdf5`

```{py:module} chemparseplot.parse.trajectory.hdf5
```

```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`load_neb_result <chemparseplot.parse.trajectory.hdf5.load_neb_result>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.load_neb_result
    :summary:
    ```
* - {py:obj}`load_neb_history <chemparseplot.parse.trajectory.hdf5.load_neb_history>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.load_neb_history
    :summary:
    ```
* - {py:obj}`result_to_profile_dat <chemparseplot.parse.trajectory.hdf5.result_to_profile_dat>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.result_to_profile_dat
    :summary:
    ```
* - {py:obj}`result_to_atoms_list <chemparseplot.parse.trajectory.hdf5.result_to_atoms_list>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.result_to_atoms_list
    :summary:
    ```
* - {py:obj}`history_to_profile_dats <chemparseplot.parse.trajectory.hdf5.history_to_profile_dats>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.history_to_profile_dats
    :summary:
    ```
* - {py:obj}`history_to_landscape_df <chemparseplot.parse.trajectory.hdf5.history_to_landscape_df>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.history_to_landscape_df
    :summary:
    ```
````

### API

````{py:function} load_neb_result(h5_file: str) -> dict
:canonical: chemparseplot.parse.trajectory.hdf5.load_neb_result

```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.load_neb_result
```
````

````{py:function} load_neb_history(h5_file: str) -> list[dict]
:canonical: chemparseplot.parse.trajectory.hdf5.load_neb_history

```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.load_neb_history
```
````

````{py:function} result_to_profile_dat(h5_file: str) -> numpy.ndarray
:canonical: chemparseplot.parse.trajectory.hdf5.result_to_profile_dat

```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.result_to_profile_dat
```
````

````{py:function} result_to_atoms_list(h5_file: str) -> list[ase.Atoms]
:canonical: chemparseplot.parse.trajectory.hdf5.result_to_atoms_list

```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.result_to_atoms_list
```
````

````{py:function} history_to_profile_dats(h5_file: str) -> list[numpy.ndarray]
:canonical: chemparseplot.parse.trajectory.hdf5.history_to_profile_dats

```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.history_to_profile_dats
```
````

````{py:function} history_to_landscape_df(h5_file: str, ira_kmax: float = 1.8)
:canonical: chemparseplot.parse.trajectory.hdf5.history_to_landscape_df

```{autodoc2-docstring} chemparseplot.parse.trajectory.hdf5.history_to_landscape_df
```
````
