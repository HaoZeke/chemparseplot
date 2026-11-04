`chemparseplot.plot.palette.assign_element_colors` gives each element one colour of a small palette.
The anchors are hydrogen, carbon, nitrogen and oxygen, the atoms of the Corey and Pauling models.
They take the colours one to one, by the pairing closest in CIEDE2000 to their usual colours.
Every other element takes the colour of the anchor nearest it in usual colour.
`render_structure` colours atoms this way by default.
`chemparseplot.plot.render.ELEMENT_COLORS` maps the 109 elements of the Jmol table in ASE to the four Ruhi colours other than coral.
