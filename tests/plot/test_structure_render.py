# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Geometry and backend choice for ``chemparseplot.plot.render``."""

import io
import logging
import math
import re
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from PIL import Image, ImageDraw, ImageFont

from chemparseplot.plot.render import (
    BOND_MIN_A,
    BOND_TOLERANCE_A,
    CANVAS,
    CAPTION_FLOOR_POINT,
    ELEMENT_COLORS,
    HBOND_BETA_MAX_DEG,
    HBOND_OO_MAX_A,
    HYDROGENS_PER_WATER,
    LABEL_POINT,
    LJ_CONTACT_RADIUS_OVER_SIGMA,
    LJ_RMIN_OVER_SIGMA,
    MM_PER_INCH,
    NATURE_SINGLE_COLUMN_MM,
    OUTLINE_POINT,
    POINTS_PER_INCH,
    PRINT_DPI,
    _caption_face,
    _caption_px,
    _caption_px_range,
    _fit_caption,
    _front_water,
    _hex_to_rgb,
    _lj_highlight,
    _paint,
    _single_frames,
    _stitch,
    _tag_print_dpi,
    _theme_hex,
    _theme_rgba,
    _water_frame,
    backend_for,
    covalent_bonds,
    covalent_radius,
    element_color,
    frame_color_overrides,
    frame_indices,
    geometric_summit,
    hydrogen_bonds,
    is_covalent_contact,
    render_structure,
    transferring_hydrogen,
)
from chemparseplot.plot.theme import (
    BATLOW_THEME,
    RUHI_COLORS,
    RUHI_THEME,
    PlotTheme,
    get_theme,
)

# Hoy and Bunker, J. Mol. Spectrosc. 74, 1-8 (1979), as tabulated by CCCBDB.
HOY_BUNKER_OH_A = 0.9578
HOY_BUNKER_ANGLE_DEG = 104.4776

# IEC 61966-2-1 (sRGB): the D65 RGB-to-XYZ matrix. CIE 15 (CIELAB): the
# linear segment of f starts at (6/29)**3.
SRGB_TO_XYZ = np.array(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
)
LAB_DELTA = 6.0 / 29.0
# WCAG 2.1 SC 1.4.11: 3:1 between a graphic and the colours next to it.
WCAG_GRAPHIC_CONTRAST = 3.0


def _pair(distance: float) -> tuple[list[str], np.ndarray]:
    return ["O", "H"], np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])


def _linear(rgb) -> np.ndarray:
    """IEC 61966-2-1 decoding of an sRGB triple in 0..1."""
    c = np.asarray(rgb, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB of an sRGB triple in 0..1, D65 white."""
    t = (SRGB_TO_XYZ @ _linear(rgb)) / SRGB_TO_XYZ.sum(axis=1)
    f = np.where(t > LAB_DELTA**3, np.cbrt(t), t / (3.0 * LAB_DELTA**2) + 4.0 / 29.0)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


def _contrast(first: str, second: str) -> float:
    """WCAG 2 contrast ratio of two colours."""
    from matplotlib.colors import to_rgb

    high, low = sorted(
        (float(SRGB_TO_XYZ[1] @ _linear(to_rgb(c))) for c in (first, second)),
        reverse=True,
    )
    return (high + 0.05) / (low + 0.05)


def _hoy_bunker_water() -> tuple[list[str], np.ndarray]:
    angle = math.radians(HOY_BUNKER_ANGLE_DEG)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [HOY_BUNKER_OH_A, 0.0, 0.0],
            [
                HOY_BUNKER_OH_A * math.cos(angle),
                HOY_BUNKER_OH_A * math.sin(angle),
                0.0,
            ],
        ]
    )
    return ["O", "H", "H"], positions


class TestLennardJones:
    def test_spheres_separated_by_the_pair_minimum_touch(self):
        distance = 2.0 ** (1.0 / 6.0)
        atoms = Atoms("He2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        frames, renderer = _single_frames([atoms], "cluster", None, None)
        assert renderer == "solvis"
        gap = distance - float(frames[0].radii[0] + frames[0].radii[1])
        assert gap == pytest.approx(0.0, abs=1e-9)

    def test_a_neighbour_beyond_the_shell_is_not_counted(self):
        near = 2.0 ** (1.0 / 6.0)
        # Past the shell midpoint from each neighbour, so the third atom is alone.
        shell = near * (1.0 + math.sqrt(2.0)) / 2.0
        far = near + shell + near / 10.0
        positions = np.array([[0.0, 0.0, 0.0], [near, 0.0, 0.0], [far, 0.0, 0.0]])
        assert _lj_highlight(positions).tolist() == [False, False, True]


class TestCanvas:
    def test_default_figure_width_is_one_column_at_print_dpi(self):
        image = Image.new("RGBA", (CANVAS, CANVAS))
        tagged = _tag_print_dpi(image, CANVAS)
        dpi = tagged.info["dpi"][0]
        width_mm = CANVAS / dpi * MM_PER_INCH
        half_pixel_mm = MM_PER_INCH / PRINT_DPI / 2.0
        assert width_mm == pytest.approx(NATURE_SINGLE_COLUMN_MM, abs=half_pixel_mm)


class TestCovalentWindow:
    def test_inside_and_outside_the_jmol_window(self):
        symbols, _ = _pair(0.0)
        limit = covalent_radius("O") + covalent_radius("H") + BOND_TOLERANCE_A
        inside = (BOND_MIN_A + limit) / 2.0
        assert is_covalent_contact(symbols, _pair(inside)[1], 0, 1)
        assert not is_covalent_contact(symbols, _pair(BOND_MIN_A / 2.0)[1], 0, 1)
        assert not is_covalent_contact(symbols, _pair(limit + BOND_MIN_A)[1], 0, 1)

    def test_gas_phase_water_has_two_bonds(self):
        symbols, positions = _hoy_bunker_water()
        assert covalent_bonds(symbols, positions) == [(0, 1), (0, 2)]


class TestHydrogenBonds:
    def test_kumar_contact_on_and_beyond_the_cutoff(self):
        symbols, positions = _hoy_bunker_water()
        # The second hydrogen sits at the Hoy-Bunker angle, which is
        # wider than the Kumar beta cutoff, so only the aligned O-H qualifies.
        assert HOY_BUNKER_ANGLE_DEG > HBOND_BETA_MAX_DEG
        on_cutoff = np.vstack([positions, [HBOND_OO_MAX_A, 0.0, 0.0]])
        found = hydrogen_bonds([*symbols, "O"], on_cutoff)
        assert found == [(1, 3)]
        beyond = np.vstack([positions, [HBOND_OO_MAX_A + 1.0, 0.0, 0.0]])
        assert hydrogen_bonds([*symbols, "O"], beyond) == []


class TestPaths:
    def test_shown_frames_are_the_ends_the_summit_and_the_midpoints(self):
        assert frame_indices(20, 9) == [0, 4, 9, 14, 19]

    def test_transferring_hydrogen_is_the_one_that_moves(self):
        symbols = ["O", "C", "H", "H"]
        first = np.array(
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.0, 0.0], [1.2, 1.0, 0.0]]
        )
        last = np.array(
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 2.0, 0.0], [1.2, 1.0, 0.0]]
        )
        assert transferring_hydrogen([(symbols, first), (symbols, last)]) == 2

    def test_geometric_summit_is_the_equidistant_frame(self):
        symbols = ["O", "H", "O"]
        frames = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        ]
        assert geometric_summit(symbols, frames, 1) == 1


class TestBackend:
    def test_snapshot_follows_the_element_count(self):
        water = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        gold = Atoms("Au4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert backend_for("snapshot", water) == "xyzrender"
        assert backend_for("snapshot", gold) == "solvis"
        assert backend_for("cluster", gold) == "solvis"
        assert backend_for("instanton", water) == "xyzrender"
        assert backend_for("neb", water) == "xyzrender"

    def test_unknown_kind(self):
        water = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        with pytest.raises(ValueError, match="Unknown structure kind"):
            backend_for("meander", water)


class TestRender:
    def test_xyzrender_snapshot_is_a_painted_image(self):
        pytest.importorskip("xyzrender")
        atoms = Atoms(
            "H2O",
            positions=_hoy_bunker_water()[1],
        )
        original = atoms.positions.copy()
        image = render_structure(atoms, kind="snapshot", canvas=80)
        assert np.array_equal(atoms.positions, original)
        assert image.dtype == np.uint8
        assert image.ndim == 3
        assert image.shape[2] == 4
        assert np.any(image[:, :, :3] != 255)

    def test_solvis_cluster_screenshot(self):
        pytest.importorskip("solvis")
        pytest.importorskip("pyvista")
        distance = LJ_RMIN_OVER_SIGMA
        atoms = Atoms("He2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        try:
            image = render_structure(atoms, kind="cluster", canvas=64)
        except (ImportError, OSError, RuntimeError) as exc:
            pytest.skip(f"solvis screenshot unavailable: {exc}")
        assert image.dtype == np.uint8
        assert image.shape[2] == 4
        assert np.any(image[:, :, :3] != 255)


class TestElementColours:
    def test_default_map_gives_each_element_its_own_ruhi_colour(self):
        values = list(ELEMENT_COLORS.values())
        assert len(set(values)) == len(values)
        assert set(values) <= set(RUHI_COLORS.values())
        # Coral is the highlight and sky the hydrogen-bond dash.
        assert RUHI_COLORS["coral"] not in values
        assert RUHI_COLORS["sky"] not in values

    def test_default_map_is_the_closest_one_to_one_pairing_with_jmol(self):
        from ase.data import atomic_numbers
        from ase.data.colors import jmol_colors

        free = [
            value for name, value in RUHI_COLORS.items() if name not in {"coral", "sky"}
        ]
        symbols = sorted(ELEMENT_COLORS)
        assert len(symbols) == len(free)

        def distance(pairing: tuple[str, ...]) -> float:
            return sum(
                float(
                    np.linalg.norm(
                        _lab(jmol_colors[atomic_numbers[symbol]])
                        - _lab(_hex_to_rgb(value))
                    )
                )
                for symbol, value in zip(symbols, pairing, strict=True)
            )

        best = min(permutations(free), key=distance)
        assert ELEMENT_COLORS == dict(zip(symbols, best, strict=True))

    def test_colour_is_an_explicit_entry(self):
        colours = {"H": "#010203", "B": "#040506", "O": "#aabbcc"}
        assert element_color("H", colours) == "#010203"
        assert element_color("B", colours) == "#040506"
        assert element_color("H", colours) != element_color("B", colours)
        with pytest.raises(KeyError, match="No body colour is set for C"):
            element_color("C", colours)
        with pytest.raises(KeyError, match="No body colour is set for N"):
            element_color("N")
        assert frame_color_overrides(["O", "H", "H"]) == {
            "O": ELEMENT_COLORS["O"],
            "H": ELEMENT_COLORS["H"],
        }
        assert frame_color_overrides(["O", "H", "H"], colours) == {
            "O": "#aabbcc",
            "H": "#010203",
        }
        with pytest.raises(KeyError, match="No body colour is set for O"):
            frame_color_overrides(["O", "H"], {"H": "#010203"})
        mask = np.zeros(2, dtype=bool)
        with pytest.raises(KeyError, match="No body colour is set for O"):
            _paint(["H", "O"], mask, None, {"H": "#010203"})
        with pytest.raises(KeyError, match="No body colour is set for N"):
            _paint(["N", "H"], mask, None)

    def test_an_element_outside_the_default_map_raises_from_render_structure(self):
        from ase.build import molecule

        with pytest.raises(KeyError, match="No body colour is set for N"):
            render_structure(molecule("NH3"), kind="snapshot", canvas=40)


def _column_point_size(image_width: int, em_px: int, width_mm: float) -> float:
    inches = width_mm / MM_PER_INCH
    return em_px * POINTS_PER_INCH / (image_width / inches)


def _em_for_caption_row(row: int) -> tuple[int, ImageFont.FreeTypeFont]:
    path = _caption_face(RUHI_THEME)
    for px in range(row, 0, -1):
        font = ImageFont.truetype(path, px)
        ascent, descent = font.getmetrics()
        if ascent + abs(descent) == row:
            return px, font
    msg = f"No caption face matches a {row} px row"
    raise AssertionError(msg)


def _drawn_caption(width: int, row: int, label: str, font) -> np.ndarray:
    ascent, _descent = font.getmetrics()
    reference = Image.new("RGBA", (width, row + 40), _theme_rgba(RUHI_THEME.facecolor))
    ImageDraw.Draw(reference).text(
        (0, ascent), label, fill=RUHI_THEME.textcolor, font=font, anchor="ls"
    )
    return np.asarray(reference)[:row]


def _px_of_caption(caption: np.ndarray, label: str) -> int:
    """The face size that redraws ``caption`` exactly, searched over every row px."""
    path = _caption_face(RUHI_THEME)
    row, width = caption.shape[:2]
    for px in range(row, 0, -1):
        font = ImageFont.truetype(path, px)
        if np.array_equal(caption, _drawn_caption(width, row, label, font)):
            return px
    msg = f"No size of the caption face redraws {label!r}"
    raise AssertionError(msg)


class TestCaption:
    def test_xyzrender_png_caption_is_at_most_seven_points(self):
        pytest.importorskip("xyzrender")
        atoms = Atoms("H2O", positions=_hoy_bunker_water()[1])
        atoms.info["label"] = "water"
        layout = 100
        image = render_structure(atoms, kind="snapshot", canvas=layout)
        panel = image.shape[1]
        zoom = panel // layout
        row = image.shape[0] - panel
        em, font = _em_for_caption_row(row)
        expected_em = LABEL_POINT * PRINT_DPI * zoom // POINTS_PER_INCH
        assert font.getlength("water") <= panel
        assert em == expected_em
        width_mm = layout / CANVAS * NATURE_SINGLE_COLUMN_MM
        printed = _column_point_size(panel, em, width_mm)
        assert printed <= LABEL_POINT
        assert np.any(image[:panel, :, :3] != 255)

    def test_five_frame_strip_caption_is_at_most_seven_points(self):
        pytest.importorskip("xyzrender")
        base = _hoy_bunker_water()[1]
        frames = []
        for step in range(5):
            positions = base.copy()
            positions[1, 0] += 0.02 * step
            frames.append(Atoms("OH2", positions=positions))
        energies = [0.0, 0.0, 1.0, 0.0, 0.0]
        image = render_structure(frames, kind="neb", energies=energies)
        count = 5
        layout = CANVAS // count
        panel = image.shape[1] // count
        assert image.shape[1] == panel * count
        zoom = panel // layout
        row = image.shape[0] - panel
        em, font = _em_for_caption_row(row)
        expected_em = LABEL_POINT * PRINT_DPI * zoom // POINTS_PER_INCH
        assert em == expected_em
        printed = _column_point_size(image.shape[1], em, NATURE_SINGLE_COLUMN_MM)
        assert printed <= LABEL_POINT
        ascent, descent = font.getmetrics()
        assert row == ascent + abs(descent)
        drawn = _drawn_caption(panel, row, "image 0", font)
        caption = image[panel : panel + row, :panel]
        assert np.array_equal(caption, drawn)
        assert np.any(drawn[em:row, :, 0] < 250)

    def test_caption_sizes_print_from_five_to_seven_points(self):
        low, high = _caption_px_range(PRINT_DPI)
        assert low * POINTS_PER_INCH / PRINT_DPI >= CAPTION_FLOOR_POINT
        assert (low - 1) * POINTS_PER_INCH / PRINT_DPI < CAPTION_FLOOR_POINT
        assert high * POINTS_PER_INCH / PRINT_DPI <= LABEL_POINT
        assert (high + 1) * POINTS_PER_INCH / PRINT_DPI > LABEL_POINT
        # At 9 dpi the band runs from 0.625 px to 0.875 px, with no whole pixel.
        with pytest.raises(ValueError, match="No whole-pixel caption size"):
            _caption_px_range(9)

    def test_five_frame_strip_steps_a_wide_label_down(self):
        pytest.importorskip("xyzrender")
        frames = [Atoms("OH2", positions=_hoy_bunker_water()[1]) for _ in range(5)]
        count = len(frames)
        layout = CANVAS // count
        probe = render_structure(frames, kind="snapshot")
        panel = probe.shape[1] // count
        assert probe.shape[1] == panel * count
        zoom = panel // layout
        dpi = PRINT_DPI * zoom
        path = _caption_face(RUHI_THEME)
        low, high = _caption_px_range(dpi)
        seven = ImageFont.truetype(path, high)
        # The shortest prefix wider than the panel at 7 pt in the installed face.
        wording = "image 12 saddle of the proton transfer path"
        label = next(
            wording[:end]
            for end in range(1, len(wording) + 1)
            if wording[end - 1] != " " and seven.getlength(wording[:end]) > panel
        )
        assert ImageFont.truetype(path, low).getlength(label) <= panel
        frames[2].info["label"] = label
        image = render_structure(frames, kind="snapshot")
        row = image.shape[0] - panel
        x0 = 2 * panel
        px = _px_of_caption(image[panel : panel + row, x0 : x0 + panel], label)
        assert CAPTION_FLOOR_POINT <= px * POINTS_PER_INCH / dpi <= LABEL_POINT
        assert ImageFont.truetype(path, px).getlength(label) <= panel
        assert ImageFont.truetype(path, px + 1).getlength(label) > panel

    def test_a_label_past_the_floor_is_ellipsised_on_a_strip(self):
        panel = 80
        dpi = PRINT_DPI
        blank = Image.new("RGBA", (panel, 20), _theme_rgba(RUHI_THEME.facecolor))
        blank.info["dpi"] = (dpi, dpi)
        label = "image 12 saddle " * 6
        text, font = _fit_caption(label, panel, float(dpi), _caption_face(RUHI_THEME))
        assert text.endswith("…")
        assert text != label
        assert font.getlength(text) <= panel
        assert font.size * POINTS_PER_INCH / dpi >= CAPTION_FLOOR_POINT
        assert (font.size - 1) * POINTS_PER_INCH / dpi < CAPTION_FLOOR_POINT
        image = _stitch([blank] * 5, [label, "a", "b", "c", "d"], RUHI_THEME)
        row = image.shape[0] - blank.height
        caption = image[blank.height : blank.height + row, :panel]
        assert np.array_equal(caption, _drawn_caption(panel, row, text, font))

    def test_png_dpi_chunk_sets_the_caption_size(self):
        image = Image.new("RGBA", (90, 90), (255, 255, 255, 255))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", dpi=(120, 120))
        buffer.seek(0)
        opened = Image.open(buffer)
        stored = float(opened.info["dpi"][0])
        # A PNG stores dpi as pixels per metre, so the round trip is not exact.
        assert abs(stored - 120) < 0.1
        tagged = _tag_print_dpi(opened, 30)
        assert float(tagged.info["dpi"][0]) == stored
        assert _caption_px(tagged) == int(LABEL_POINT * stored / POINTS_PER_INCH)

    def test_partial_element_map_raises_from_render_structure(self):
        atoms = Atoms("H2O", positions=_hoy_bunker_water()[1])
        with pytest.raises(KeyError, match="No body colour is set for O"):
            render_structure(
                atoms,
                kind="snapshot",
                canvas=40,
                element_colors={"H": "#010203"},
            )

    def test_theme_override_changes_the_caption_ink(self):
        pytest.importorskip("xyzrender")
        atoms = Atoms("H2O", positions=_hoy_bunker_water()[1])
        atoms.info["label"] = "water"
        theme = get_theme("ruhi", textcolor="#ff0000")
        image = render_structure(atoms, kind="snapshot", canvas=80, theme=theme)
        panel = image.shape[1]
        band = image[panel:, :, :3]
        red = (band[:, :, 0] > 200) & (band[:, :, 1] < 40) & (band[:, :, 2] < 40)
        assert np.any(red)

    def test_missing_caption_font_falls_back_and_names_the_face(self, caplog):
        pytest.importorskip("xyzrender")
        from matplotlib.font_manager import FontProperties, findfont

        atoms = Atoms("H2O", positions=_hoy_bunker_water()[1])
        atoms.info["label"] = "water"
        theme = get_theme("ruhi", caption_font="No Such Caption Face")
        with caplog.at_level(logging.WARNING, logger="chemparseplot.plot.render"):
            face = _caption_face(theme)
            image = render_structure(atoms, kind="snapshot", canvas=40, theme=theme)
        assert face == findfont(FontProperties(family=["sans-serif"]))
        assert "No Such Caption Face" in caplog.text
        assert image.shape[0] > image.shape[1]

    def test_caption_font_defaults_when_omitted(self):
        theme = PlotTheme(
            name="bare",
            font_family="DejaVu Sans",
            font_size=12,
            facecolor="white",
            textcolor="black",
            edgecolor="black",
            gridcolor="white",
            cmap_profile="ruhi_diverging",
            cmap_landscape="ruhi_diverging",
            highlight_color="black",
        )
        assert theme.caption_font == RUHI_THEME.caption_font
        assert BATLOW_THEME.caption_font == RUHI_THEME.caption_font
        assert RUHI_THEME.caption_font == RUHI_THEME.font_family


class TestOutline:
    def test_every_xyzrender_outline_is_the_ink_on_a_deep_figure(self, monkeypatch):
        xyzrender = pytest.importorskip("xyzrender")
        original = xyzrender.render
        svgs = []

        def capture(xyz, *args, output=None, **kwargs):
            svg = Path(output).with_suffix(".svg")
            original(xyz, *args, output=svg, **kwargs)
            svgs.append(svg.read_text())
            return original(xyz, *args, output=output, **kwargs)

        monkeypatch.setattr(xyzrender, "render", capture)
        # Waters on the corners of a tetrahedron about 17 A on a side: no
        # hydrogen bonds, and several angstroms of depth in any orientation,
        # which is where paton's fog would lighten an outline.
        corners = 6.0 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        water = _hoy_bunker_water()[1]
        positions = np.vstack([water + corner for corner in corners])
        atoms = Atoms("OH2" * len(corners), positions=positions)
        render_structure(atoms, kind="snapshot", canvas=200)
        assert len(svgs) == 1
        strokes = re.findall(r'<circle [^>]*stroke="(#[0-9a-fA-F]{6})"', svgs[0])
        ink = _theme_hex(RUHI_THEME.textcolor)
        assert len(strokes) == len(atoms)
        assert {stroke.lower() for stroke in strokes} == {ink}
        assert _contrast(ink, RUHI_THEME.facecolor) >= WCAG_GRAPHIC_CONTRAST

    def test_cluster_atoms_carry_an_ink_outline(self, monkeypatch):
        pytest.importorskip("solvis")
        pyvista = pytest.importorskip("pyvista")
        meshes = []
        original = pyvista.Plotter.add_mesh

        def record(self, mesh, *args, **kwargs):
            meshes.append((mesh, kwargs))
            return original(self, mesh, *args, **kwargs)

        monkeypatch.setattr(pyvista.Plotter, "add_mesh", record)
        distance = LJ_RMIN_OVER_SIGMA
        atoms = Atoms("He2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        canvas = 64
        render_structure(atoms, kind="cluster", canvas=canvas)
        hulls = [(mesh, kw) for mesh, kw in meshes if kw.get("culling") == "front"]
        bodies = [mesh for mesh, kw in meshes if kw.get("culling") != "front"]
        assert len(hulls) == len(bodies) == len(atoms)
        span = distance + 2.0 * LJ_CONTACT_RADIUS_OVER_SIGMA
        outline_px = OUTLINE_POINT * PRINT_DPI / POINTS_PER_INCH
        for body, (hull, kwargs) in zip(bodies, hulls, strict=True):
            assert hull.center == pytest.approx(body.center)
            # pyvista puts the sphere poles on z, so the z extent is exact.
            delta = (
                hull.bounds[5] - hull.bounds[4] - body.bounds[5] + body.bounds[4]
            ) / 2
            # The view is the span plus one outline on each side.
            assert delta * canvas / (span + 2.0 * delta) == pytest.approx(outline_px)
            assert kwargs["color"] == _theme_hex(RUHI_THEME.textcolor)
            assert kwargs["lighting"] is False
        assert _contrast(kwargs["color"], RUHI_THEME.facecolor) >= WCAG_GRAPHIC_CONTRAST
        with pytest.raises(ValueError, match="wider than two outlines"):
            render_structure(atoms, kind="cluster", canvas=2)


class TestPeriodicWater:
    def test_synthetic_cell_wraps_an_interior_hbond_and_a_ghost_oxygen(self):
        # Synthetic Hoy-Bunker waters. The edge is the shortest orthorhombic
        # length on which a contact at the Kumar cutoff is not also its own
        # other image at that cutoff. The drawn contact sits one bond-minimum
        # inside the cutoff, so the case is not the equality.
        edge = 2.0 * HBOND_OO_MAX_A + BOND_MIN_A
        contact = HBOND_OO_MAX_A - BOND_MIN_A
        assert contact < HBOND_OO_MAX_A
        angle = math.radians(HOY_BUNKER_ANGLE_DEG)
        donor = np.array([edge - BOND_MIN_A, 0.0, 0.0])
        donor_h = donor + np.array([HOY_BUNKER_OH_A, 0.0, 0.0])
        donor_h2 = donor + HOY_BUNKER_OH_A * np.array(
            [math.cos(angle), math.sin(angle), 0.0]
        )
        # The acceptor starts one cell past the boundary. Its covalent
        # geometry is intact, so the wrapper has to bring it back.
        acceptor = donor + np.array([contact + edge, 0.0, 0.0])
        acceptor_h = acceptor + np.array([0.0, HOY_BUNKER_OH_A, 0.0])
        acceptor_h2 = acceptor + HOY_BUNKER_OH_A * np.array(
            [0.0, math.cos(angle), math.sin(angle)]
        )
        assert acceptor[0] > edge
        symbols = ["O", "H", "H", "O", "H", "H"]
        positions = np.vstack(
            [donor, donor_h, donor_h2, acceptor, acceptor_h, acceptor_h2]
        )
        atoms = Atoms(symbols, positions=positions, cell=[edge, edge, edge], pbc=True)
        frame = _water_frame(atoms, None, "")
        assert frame is not None
        assert frame.symbols == [*symbols, "O"]
        ghost = len(frame.symbols) - 1
        assert frame.hbonds == [(1, ghost)]
        assert len(frame.bonds) == HYDROGENS_PER_WATER * 2
        assert all(ghost not in pair for pair in frame.bonds)
        assert not frame.highlight[ghost]
        donor_to_ghost = np.linalg.norm(frame.positions[ghost] - frame.positions[0])
        assert donor_to_ghost == pytest.approx(contact)
        assert donor_to_ghost < HBOND_OO_MAX_A
        ghost_shift = frame.positions[ghost] - frame.positions[3]
        assert ghost_shift == pytest.approx([edge, 0.0, 0.0])
        drawn_oh = np.linalg.norm(frame.positions[1] - frame.positions[0])
        assert drawn_oh == pytest.approx(HOY_BUNKER_OH_A)

    def test_default_highlight_is_the_water_in_view(self):
        # Two waters stacked along c at the cell centre, and one beside
        # them. The lower one is nearest the centre, but the one above it
        # covers its oxygen in the image.
        edge = 2.0 * HBOND_OO_MAX_A + BOND_MIN_A
        gap = HBOND_OO_MAX_A - BOND_MIN_A
        centre = np.full(3, edge / 2.0)
        up = np.array([0.0, 0.0, gap])
        side = np.array([gap, 0.0, 0.0])
        symbols, water = _hoy_bunker_water()
        sites = [centre, centre + up, centre - side]
        positions = np.vstack([water + site for site in sites])
        atoms = Atoms(symbols * 3, positions=positions, cell=[edge] * 3, pbc=True)
        frame = _water_frame(atoms, None, "")
        assert frame is not None
        assert np.flatnonzero(frame.highlight).tolist() == [3, 4, 5]

    def test_every_water_is_a_candidate_when_every_oxygen_is_covered(self):
        symbols, water = _hoy_bunker_water()
        gap = HBOND_OO_MAX_A - BOND_MIN_A
        up = np.array([0.0, 0.0, gap])
        sites = [np.zeros(3), np.array([gap, 0.0, 0.0])]
        # A ghost oxygen sits straight above each water.
        ghosts = [site + up for site in sites]
        positions = np.vstack([*(water + site for site in sites), *ghosts])
        symbols = symbols * 2 + ["O", "O"]
        radii = np.array([covalent_radius(symbol) for symbol in symbols])
        molecules = [(0, 1, 2), (3, 4, 5)]
        chosen = _front_water(molecules, symbols, positions, radii, sites[1])
        assert chosen == (3, 4, 5)
