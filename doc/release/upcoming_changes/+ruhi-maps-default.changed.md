The default colour maps are Ruhi maps.
`ruhi_seq` replaces viridis in the nudged elastic band and optimization landscapes and their path overlays.
It also replaces coolwarm in `plot_mmf_peaks_overlay`, which takes `cmap`.
It replaces the `ruhi_diverging` blend, whose lightness is not monotone, in the theme and the ChemGP surfaces.
`chemparseplot.plot.landscape.plot_landscape_surface` uses `ruhi_seq` in place of that blend too.
Earlier bands in `plot_neb_evolution` and Wiberg bond orders use only the part of `ruhi_seq` with a 3:1 contrast on the background.
`mark_saddle_point` draws its marker in a Ruhi colour.
`plot_orca_neb_profile` and `plot_orca_neb_energy_profile` mark their key points in Ruhi colours.
`plot_orca_neb_energy_profile` also draws its line in the first Ruhi line colour by default.
`setup_global_theme` sets the Ruhi line colours as the colour cycle and the theme's map as `image.cmap`.
