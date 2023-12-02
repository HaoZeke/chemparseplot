# Contributing tutorials

The documentation for the `tutorials` mimics that of `numpy-tutorials` to a
large extent. Since we use `pdm` as the management system, we define the
required development dependencies in the `-dG nbdoc` group.

```bash
pdm run jupyter lab --ServerApp.allow_remote_access=1 \
    --ServerApp.open_browser=False --port=8889
```
