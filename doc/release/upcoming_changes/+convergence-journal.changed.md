`plot_single_ended_convergence` draws in the journal style, with panel letters.
The convergence measure and the step size are on log scales.
Lines are labelled at their ends or in a legend off the data, and `threshold` draws a dashed, labelled threshold.
`plot_convergence_panel` puts the step size on a log scale too and marks only the last iteration.
`plot_single_ended_profile` gives each trajectory a Ruhi colour and its own line type.
It takes line width and marker size from the rcParams, where they were fixed numbers.
The line types are those of the drawing standard ISO 128-2 of the International Organization for Standardization (ISO).
`save_standard_figure` saves constrained-layout figures at their exact size.
