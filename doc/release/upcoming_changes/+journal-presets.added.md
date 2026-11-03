`chemparseplot.plot.journal` sizes and styles figures for five journals.
The `acs` preset covers the Journal of Chemical Theory and Computation and the Journal of Chemical Information and Modeling.
`elsevier` covers Computer Physics Communications and `rsc` Physical Chemistry Chemical Physics.
`aip` covers The Journal of Chemical Physics.
`journal_figure` makes a figure of the exact single- or double-column width.
Its panels are half again as wide as they are tall unless the data give them a shape.
Text is set at the body, `\small` and `\footnotesize` sizes of each journal's two-column page, in the theme's face.
For `ruhi` that face is Atkinson Hyperlegible, and the faces the journal names stand in where it is not installed.
Line widths, tick lengths, gaps and marker sizes follow from the text size.
They follow the lettering standard ISO 3098-1 and the line standard ISO 128-2 of the International Organization for Standardization (ISO).
`series_style` pairs each series' Ruhi colour with a line type from `LINE_TYPES`, where the ISO 128-2 types with one kind of mark come before those with two.
`label_panels` letters the panels, and `label_lines` and `label_hline` label lines at their ends.
`smart_legend` places a legend where it covers no data.
`sci_axis` writes numbers from 0.1 to 1000 plain.
`relative_error` and `log_error_axis` put relative errors on log scales.
`save_figure` writes PDF, SVG and raster files at the exact figure size, with TrueType fonts embedded and no timestamps.
`setup_publication_theme` takes `journal` to apply the same style.
