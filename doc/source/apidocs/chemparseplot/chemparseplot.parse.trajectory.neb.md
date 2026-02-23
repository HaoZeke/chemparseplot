# {py:mod}`chemparseplot.parse.trajectory.neb`

```{py:module} chemparseplot.parse.trajectory.neb
```

```{autodoc2-docstring} chemparseplot.parse.trajectory.neb
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`load_trajectory <chemparseplot.parse.trajectory.neb.load_trajectory>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.load_trajectory
    :summary:
    ```
* - {py:obj}`compute_cumulative_distance <chemparseplot.parse.trajectory.neb.compute_cumulative_distance>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.compute_cumulative_distance
    :summary:
    ```
* - {py:obj}`compute_tangent_force <chemparseplot.parse.trajectory.neb.compute_tangent_force>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.compute_tangent_force
    :summary:
    ```
* - {py:obj}`extract_profile_data <chemparseplot.parse.trajectory.neb.extract_profile_data>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.extract_profile_data
    :summary:
    ```
* - {py:obj}`trajectory_to_profile_dat <chemparseplot.parse.trajectory.neb.trajectory_to_profile_dat>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.trajectory_to_profile_dat
    :summary:
    ```
* - {py:obj}`trajectory_to_landscape_df <chemparseplot.parse.trajectory.neb.trajectory_to_landscape_df>`
  - ```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.trajectory_to_landscape_df
    :summary:
    ```
````

### API

````{py:function} load_trajectory(traj_file: str) -> list[ase.Atoms]
:canonical: chemparseplot.parse.trajectory.neb.load_trajectory

```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.load_trajectory
```
````

````{py:function} compute_cumulative_distance(atoms_list: list[ase.Atoms]) -> numpy.ndarray
:canonical: chemparseplot.parse.trajectory.neb.compute_cumulative_distance

```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.compute_cumulative_distance
```
````

````{py:function} compute_tangent_force(atoms_list: list[ase.Atoms], energies: numpy.ndarray) -> numpy.ndarray
:canonical: chemparseplot.parse.trajectory.neb.compute_tangent_force

```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.compute_tangent_force
```
````

````{py:function} extract_profile_data(atoms_list: list[ase.Atoms]) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
:canonical: chemparseplot.parse.trajectory.neb.extract_profile_data

```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.extract_profile_data
```
````

````{py:function} trajectory_to_profile_dat(atoms_list: list[ase.Atoms]) -> numpy.ndarray
:canonical: chemparseplot.parse.trajectory.neb.trajectory_to_profile_dat

```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.trajectory_to_profile_dat
```
````

````{py:function} trajectory_to_landscape_df(atoms_list: list[ase.Atoms], ira_kmax: float = 1.8, step: int = 0)
:canonical: chemparseplot.parse.trajectory.neb.trajectory_to_landscape_df

```{autodoc2-docstring} chemparseplot.parse.trajectory.neb.trajectory_to_landscape_df
```
````
