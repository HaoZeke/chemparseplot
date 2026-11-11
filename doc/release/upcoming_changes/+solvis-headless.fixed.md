A cluster render starts Xvfb when the process has no display.
The VTK wheel draws through GLX.
Without a display, EGL, or OSMesa, the screenshot segfaulted.
