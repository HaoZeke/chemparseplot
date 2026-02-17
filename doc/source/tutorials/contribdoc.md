# Contributing tutorials

The documentation for the `tutorials` mimics that of `numpy-tutorials` to a
large extent. The required notebook dependencies are in the `nbdoc` optional
dependency group.

```bash
uv run jupyter lab --ServerApp.allow_remote_access=1 \
    --ServerApp.open_browser=False --port=8889
```
