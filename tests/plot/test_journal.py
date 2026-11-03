# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Journal presets, the derived figure style, labelling and saving."""

import math
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font, LoadFlags
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.textpath import TextToPath
from matplotlib.ticker import LogFormatterSciNotation, NullFormatter

from chemparseplot.plot import journal as jr
from chemparseplot.plot.palette import (
    RUHI_COLORS,
    RUHI_CYCLE,
    RUHI_SEQUENTIAL,
    WCAG_GRAPHICS_CONTRAST,
    WCAG_TEXT_CONTRAST,
    contrast_ratio,
)
from chemparseplot.plot.theme import get_theme, setup_publication_theme

ROUND_OFF = math.sqrt(np.finfo(float).eps)
FACE = "DejaVu Sans"

# Lengths as each guideline states them, in its own units. Where two are
# given they are rounded conversions of each other and the preset takes the
# smaller, so a figure fits both.
STATED = {
    "acs": {
        "single": (jr.bp(240), 3.33),
        "double": (jr.bp(504), 7.0),
        "max_height": (jr.bp(660), 9.167),
    },
    "aip": {
        "single": (3.37, jr.cm(8.5)),
        "double": (6.69, jr.cm(17.0)),
        "max_height": (8.25, jr.cm(21.1)),
    },
    "elsevier": {
        "single": (jr.mm(90), jr.bp(255)),
        "onehalf": (jr.mm(140), jr.bp(397)),
        "double": (jr.mm(190), jr.bp(539)),
    },
    "rsc": {
        "single": (jr.cm(8.3),),
        "double": (jr.cm(17.1),),
        "max_height": (jr.cm(23.3),),
    },
}

# ISO 3098-1:2015 Table 2, lettering type B, as multiples of the capital
# height h; ISO 128-2:2020 line widths as multiples of the narrow width.
ISO_3098_B = {
    "stroke": 1 / 10,
    "x_height": 7 / 10,
    "descender": 3 / 10,
    "char_space": 2 / 10,
    "word_space": 6 / 10,
    "baseline_pitch": 15 / 10,
}
ISO_128_RATIOS = {"wide": 2.0, "extra": 4.0}
ISO_128_ELEMENTS = {"dot": 0.5, "gap": 3.0, "dash": 12.0, "long_dash": 24.0}

# matplotlib writes PDF reals to ten decimals.
PDF_REAL_TOL = 0.5e-10


@pytest.fixture(autouse=True)
def clean_rc():
    with mpl.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def dejavu(monkeypatch):
    monkeypatch.setattr(jr, "resolve_font", lambda candidates: FACE)


def _style(key):
    return jr.figure_style(key, font=FACE)


class TestPresets:
    @pytest.mark.parametrize("key", sorted(STATED))
    def test_lengths_are_the_smaller_stated_value(self, key):
        preset = jr.get_preset(key)
        for what, lengths in STATED[key].items():
            value = preset.max_height if what == "max_height" else preset.width(what)
            assert value == min(lengths)
        assert set(preset.widths) == {k for k in STATED[key] if k != "max_height"}

    def test_elsevier_states_no_height(self):
        assert jr.ELSEVIER.max_height is None

    @pytest.mark.parametrize(
        ("alias", "key"),
        [
            ("jctc", "acs"),
            ("JCIM", "acs"),
            ("jcp", "aip"),
            ("cpc", "elsevier"),
            ("pccp", "rsc"),
        ],
    )
    def test_aliases(self, alias, key):
        assert jr.get_preset(alias) is jr.PRESETS[key]

    def test_lookup_errors(self):
        assert jr.get_preset(jr.RSC) is jr.RSC
        with pytest.raises(ValueError, match="Unknown journal"):
            jr.get_preset("nature")
        with pytest.raises(ValueError, match="no 'onehalf' width"):
            jr.ACS.width("onehalf")

    def test_default_is_acs(self):
        assert jr.get_preset() is jr.ACS


class TestUnits:
    def test_conversions(self):
        assert jr.mm(25.4) == 1.0
        assert jr.cm(2.54) == 1.0
        assert jr.bp(72.0) == 1.0
        assert jr.tex_pt(72.27) == 1.0
        assert jr.tex_pt_to_bp(72.27) == pytest.approx(72.0, rel=ROUND_OFF)
        assert jr.bp_to_tex_pt(jr.tex_pt_to_bp(10.0)) == pytest.approx(
            10.0, rel=ROUND_OFF
        )


class TestFonts:
    def test_cap_height_matches_the_font_file(self):
        path = font_manager.findfont(FACE, fallback_to_default=False)
        font = FT2Font(path)
        glyph = font.load_char(ord("H"), flags=LoadFlags.NO_SCALE)
        expected = glyph.bbox[3] / font.units_per_EM
        # FreeType outlines are in 1/64 pixel at TextToPath.FONT_SCALE pixels
        # per em, so the measured height is within half of that unit.
        tol = 0.5 / (64 * TextToPath.FONT_SCALE)
        assert jr.cap_height(FACE) == pytest.approx(expected, abs=tol)

    def test_resolve_font_falls_back_to_dejavu(self):
        assert jr.resolve_font(["No Such Face"]) == FACE
        assert jr.resolve_font([FACE]) == FACE

    @pytest.mark.parametrize("key", sorted(jr.PRESETS))
    def test_resolved_faces_are_truetype(self, key):
        face = jr.figure_style(key).font
        path = font_manager.findfont(face, fallback_to_default=False)
        assert path.lower().endswith((".ttf", ".ttc"))

    @pytest.mark.parametrize("key", sorted(jr.PRESETS))
    def test_theme_face_by_default(self, key):
        face = get_theme("ruhi").font_family
        if jr.resolve_font([face]) != face:
            pytest.skip(f"{face} is not installed")
        assert jr.figure_style(key).font == face

    def test_journal_faces_stand_in_for_a_missing_theme_face(self):
        theme = get_theme("ruhi", font_family="No Such Face")
        assert jr.figure_style("acs", theme=theme).font == jr.resolve_font(jr.ACS.fonts)


class TestStyle:
    @pytest.mark.parametrize("key", sorted(jr.PRESETS))
    def test_text_sizes_follow_the_class(self, key):
        preset = jr.PRESETS[key]
        style = _style(key)
        expected = [max(jr.tex_pt_to_bp(s), preset.min_text) for s in preset.class_sizes]
        assert [style.body, style.small, style.footnote] == pytest.approx(expected)
        assert min(style.body, style.small, style.footnote) >= preset.min_text

    @pytest.mark.parametrize(
        ("key", "sizes"),
        [
            # The published ACS page, in PostScript points.
            ("acs", (10.0, 9.0, 8.0)),
            # revtex and the RSC class set 8 and 7 TeX points, below the
            # journals' lettering minima of 8 and 7 PostScript points.
            ("aip", (jr.tex_pt_to_bp(10.0), jr.tex_pt_to_bp(9.0), 8.0)),
            ("rsc", (jr.tex_pt_to_bp(9.0), jr.tex_pt_to_bp(8.0), 7.0)),
        ],
    )
    def test_text_sizes_in_points(self, key, sizes):
        style = _style(key)
        assert (style.body, style.small, style.footnote) == pytest.approx(
            sizes, rel=ROUND_OFF
        )

    @pytest.mark.parametrize("key", sorted(jr.PRESETS))
    def test_lengths_follow_iso_3098_and_128(self, key):
        style = _style(key)
        h = style.cap
        assert h == pytest.approx(style.body * jr.cap_height(FACE))
        narrow = ISO_3098_B["stroke"] * h / math.sqrt(2.0)
        assert style.narrow == pytest.approx(max(narrow, style.preset.min_line or 0.0))
        assert style.wide == pytest.approx(ISO_128_RATIOS["wide"] * style.narrow)
        assert style.extra == pytest.approx(ISO_128_RATIOS["extra"] * style.narrow)
        assert style.tick == pytest.approx(ISO_3098_B["x_height"] * h)
        assert style.minor_tick == pytest.approx(ISO_3098_B["descender"] * h)
        assert style.pad == pytest.approx(ISO_3098_B["word_space"] * h)
        assert style.gap == pytest.approx(ISO_3098_B["char_space"] * h)
        assert style.marker == pytest.approx(h)
        assert style.marker_large == pytest.approx(math.sqrt(2.0) * h)
        assert style.pitch == pytest.approx(
            ISO_3098_B["baseline_pitch"] * h * style.small / style.body
        )

    def test_journal_minimum_line_binds(self):
        # In the theme's face the derived narrow width for ACS falls below
        # its 0.5 pt minimum.
        face = get_theme("ruhi").font_family
        style = jr.figure_style("acs", font=face)
        if style.font != face:
            pytest.skip(f"{face} is not installed")
        assert ISO_3098_B["stroke"] * style.cap / math.sqrt(2.0) < jr.ACS.min_line
        assert style.narrow == jr.ACS.min_line

    def test_rc(self):
        style = _style("acs")
        rc = style.rc()
        assert rc["font.size"] == rc["axes.labelsize"] == style.body
        assert rc["xtick.labelsize"] == rc["legend.fontsize"] == style.small
        assert rc["lines.linewidth"] == style.wide
        assert rc["axes.linewidth"] == rc["xtick.major.width"] == style.narrow
        assert rc["xtick.major.size"] == style.tick
        assert rc["savefig.dpi"] == jr.ACS.raster_dpi
        assert rc["pdf.fonttype"] == rc["ps.fonttype"] == 42
        assert rc["savefig.bbox"] == "standard"
        assert rc["image.cmap"] == RUHI_SEQUENTIAL
        assert rc["axes.prop_cycle"].by_key()["color"] == list(RUHI_CYCLE)
        assert rc["axes.formatter.limits"] == list(jr.SI_POWERLIMITS)
        assert rc["font.sans-serif"][0] == FACE

    def test_apply_and_context(self, dejavu):
        before = mpl.rcParams["font.size"]
        with jr.journal_style("rsc") as style:
            assert mpl.rcParams["font.size"] == style.body
        assert mpl.rcParams["font.size"] == before
        style = jr.apply_style("jcp")
        assert mpl.rcParams["font.size"] == style.body
        assert mpl.rcParams["savefig.dpi"] == jr.AIP.raster_dpi

    def test_publication_theme_takes_a_journal(self, dejavu):
        setup_publication_theme(get_theme("ruhi"), journal="jctc")
        assert mpl.rcParams["font.size"] == pytest.approx(10.0)
        assert mpl.rcParams["savefig.bbox"] != "tight"

    def test_halo(self):
        style = _style("acs")
        (stroke,) = style.halo("white")
        assert stroke._gc["linewidth"] == style.wide
        assert stroke._gc["foreground"] == "white"

    def test_iso_128_line_types(self):
        codes = {code: pattern for code, _, pattern in jr.LINE_TYPES}
        e = ISO_128_ELEMENTS
        assert codes["01"] == "solid"
        assert codes["02"] == (0, (e["dash"], e["gap"]))
        assert codes["07"] == (0, (e["dot"], e["gap"]))
        assert codes["10"] == (0, (e["dash"], e["gap"], e["dot"], e["gap"]))
        assert codes["04"] == (0, (e["long_dash"], e["gap"], e["dot"], e["gap"]))


class TestFigures:
    def test_journal_figure_size(self, dejavu):
        fig, ax = jr.journal_figure("acs")
        width, height = fig.get_size_inches()
        assert width == jr.ACS.width("single")
        # Tufte (1983), p. 190: "about 50 percent wider than tall".
        assert width == pytest.approx(1.5 * height)
        fig, axes = jr.journal_figure("acs", "double", nrows=2, ncols=2, aspect=0.5)
        assert tuple(fig.get_size_inches()) == pytest.approx((7.0, 3.5))
        assert axes.shape == (2, 2)

    def test_journal_figure_height_limit(self, dejavu):
        with pytest.raises(ValueError, match="exceeds"):
            jr.journal_figure("acs", height=jr.ACS.max_height * 2)

    def test_given_style_leaves_rcparams(self, dejavu):
        before = mpl.rcParams["font.size"]
        jr.journal_figure(style=_style("rsc"))
        assert mpl.rcParams["font.size"] == before

    def test_style_leaves_other_figures_to_their_own_layout(self, dejavu):
        style = jr.apply_style("acs")
        fig, _ = jr.journal_figure(style=style)
        assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)
        other, ax = plt.subplots()
        other.colorbar(ax.imshow([[0.0, 1.0]]))
        other.tight_layout()
        assert not isinstance(other.get_layout_engine(), ConstrainedLayoutEngine)

    def test_fit_height_removes_empty_bands(self, dejavu):
        fig, ax = jr.journal_figure("acs", height=6.0)
        ax.plot([0, 1], [0, 1])
        ax.set_aspect("equal")
        dpi = jr.ACS.raster_dpi
        height = jr.fit_height(fig, dpi=dpi)
        assert fig.get_size_inches()[0] == jr.ACS.width("single")
        assert height < 6.0
        assert jr.fit_height(fig, dpi=dpi) == pytest.approx(height, abs=1.0 / dpi)


class TestSaving:
    def test_pdf_page_is_the_column(self, dejavu, tmp_path):
        fig, ax = jr.journal_figure("acs")
        ax.plot([0, 1], [0, 1])
        path = jr.save_figure(fig, tmp_path / "f.pdf")
        box = re.search(rb"/MediaBox \[ *0 0 ([\d.]+) ([\d.]+) *\]", path.read_bytes())
        width, height = fig.get_size_inches() * jr.BP_PER_INCH
        assert float(box.group(1)) == pytest.approx(width, abs=PDF_REAL_TOL)
        assert float(box.group(2)) == pytest.approx(height, abs=PDF_REAL_TOL)

    @pytest.mark.parametrize("ext", ["pdf", "svg"])
    def test_same_bytes_every_run(self, dejavu, tmp_path, ext):
        def run(name):
            fig, ax = jr.journal_figure("acs")
            ax.plot([0, 1], [0, 1], label="a")
            ax.set_xlabel("Path length (Å)")
            data = jr.save_figure(fig, tmp_path / f"{name}.{ext}").read_bytes()
            plt.close(fig)
            return data

        assert run("a") == run("b")

    def test_fonts_embedded_as_truetype(self, dejavu, tmp_path):
        fig, ax = jr.journal_figure("acs")
        ax.set_xlabel(r"$\Delta E^{\ddagger}$ (eV)")
        data = jr.save_figure(fig, tmp_path / "f.pdf").read_bytes()
        assert b"/FontFile2" in data
        assert b"/Type3" not in data


class TestPanelLetters:
    def test_letters_share_the_column_edge(self, dejavu):
        fig, axes = jr.journal_figure("acs", "double", nrows=2, ncols=2)
        axes[0, 0].plot([0, 1], [0, 1e4])
        axes[1, 0].plot([0, 1], [0, 1])
        axes[0, 1].plot([0, 1], [0, 1])
        axes[1, 1].plot([0, 1], [0, 1e4])
        texts = jr.label_panels(fig)
        assert [t.get_text() for t in texts] == ["(a)", "(b)", "(c)", "(d)"]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer) for t in texts]
        assert boxes[0].x0 == pytest.approx(boxes[2].x0, abs=ROUND_OFF)
        assert boxes[1].x0 == pytest.approx(boxes[3].x0, abs=ROUND_OFF)
        for ax, box in zip(axes.flat, boxes, strict=True):
            assert box.x0 <= ax.get_tightbbox(renderer).x0 + ROUND_OFF
            assert box.y0 >= ax.get_window_extent(renderer).y1

    def test_format_and_labels(self, dejavu):
        fig, axes = jr.journal_figure("acs", ncols=2)
        texts = jr.label_panels(axes, ["i", "ii"], fmt="{}.")
        assert [t.get_text() for t in texts] == ["i.", "ii."]


class TestSpread:
    def test_feasible_positions_are_kept(self):
        y = np.array([0.0, 2.0, 5.0])
        np.testing.assert_array_equal(jr._spread(y, 1.0), y)

    @pytest.mark.parametrize(
        ("y", "expected"),
        [([0.0, 0.0], [-0.5, 0.5]), ([0.0, 0.0, 0.0], [-1.0, 0.0, 1.0])],
    )
    def test_least_squares(self, y, expected):
        np.testing.assert_allclose(jr._spread(np.array(y), 1.0), expected)

    def test_spacing_and_mean(self):
        y = np.sort(np.random.default_rng(1).normal(size=40))
        pitch = 0.3
        out = jr._spread(y, pitch)
        assert np.all(np.diff(out) >= pitch * (1 - ROUND_OFF))
        assert out.mean() == pytest.approx(y.mean())


class TestDirectLabels:
    def test_labels_end_lines_apart_and_readable(self, dejavu):
        style = jr.apply_style("acs")
        fig, ax = jr.journal_figure(style=style)
        x = np.linspace(0, 1, 20)
        ax.plot(x, x, label="a", color=RUHI_COLORS["sunshine"])
        ax.plot(x, x + 1e-3, label="b")
        ax.plot(x, -x, label="_hidden")
        texts = jr.direct_labels(ax)
        assert [t.get_text() for t in texts] == ["a", "b"]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        centres = [t.get_window_extent(renderer).y0 for t in texts]
        pitch_px = style.pitch * fig.dpi / jr.BP_PER_INCH
        # Each line end moves by less than one output pixel after its label
        # is placed, so two labels lose at most two of the pitch.
        output_px = fig.dpi / mpl.rcParams["savefig.dpi"]
        assert abs(centres[1] - centres[0]) >= pitch_px - 2 * output_px
        for t in texts:
            assert contrast_ratio(t.get_color(), ax.get_facecolor()) >= WCAG_TEXT_CONTRAST

    @staticmethod
    def _crossed(ax):
        """A labelled line whose end sits on an unlabelled one."""
        ax.plot([0.0, 0.5], [0.0, 0.5], label="up")
        ax.plot([0.0, 1.0], [0.5, 0.5], label="_flat")
        ax.set_xlim(0, 1)
        return ax.get_lines()[:1]

    def test_label_lines_falls_back_to_a_legend(self, dejavu):
        fig, ax = jr.journal_figure("acs")
        out = jr.label_lines(ax, self._crossed(ax))
        assert len(out) == 1
        assert isinstance(out[0], Legend)
        assert not [t for t in ax.texts if t.get_text() == "up"]

    def test_label_lines_without_fallback(self, dejavu):
        fig, ax = jr.journal_figure("acs")
        assert jr.label_lines(ax, self._crossed(ax), fallback=False) == []
        assert ax.get_legend() is None
        assert not ax.texts

    def test_label_lines_keeps_free_labels(self, dejavu):
        fig, ax = jr.journal_figure("acs")
        x = np.linspace(0, 0.6, 20)
        ax.plot(x, x, label="up")
        ax.plot(x, 1 - x, label="down")
        ax.set_xlim(0, 1)
        out = jr.label_lines(ax)
        assert sorted(t.get_text() for t in out) == ["down", "up"]
        assert ax.get_legend() is None


class TestSeries:
    def test_styles_pass_contrast_and_repeat_late(self):
        colors = {
            c for c in RUHI_CYCLE if contrast_ratio(c, "white") >= WCAG_GRAPHICS_CONTRAST
        }
        period = math.lcm(len(colors), len(jr.LINE_TYPES))
        styles = [jr.series_style(i) for i in range(period + 1)]
        pairs = [(s["color"], str(s["linestyle"])) for s in styles]
        assert len(set(pairs[:period])) == period
        assert pairs[period] == pairs[0]
        assert {s["color"] for s in styles} == colors


class TestLegends:
    def test_inside_order_is_matplotlibs(self):
        codes = sorted(Legend.codes.items(), key=lambda item: item[1])
        places = ("center right" if loc == "right" else loc for loc, _ in codes[1:])
        assert jr._INSIDE == tuple(dict.fromkeys(places))

    def test_inside_where_free(self, dejavu):
        fig, ax = jr.journal_figure("acs")
        ax.plot([0, 0.2], [0, 0.2], label="a")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        legend = jr.smart_legend(ax)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = legend.get_window_extent(renderer)
        assert ax.get_window_extent(renderer).contains(box.x0, box.y0)
        assert not jr._overlaps_data(ax, box, renderer)

    def test_above_when_data_fills_the_axes(self, dejavu):
        fig, ax = jr.journal_figure("acs")
        ax.imshow(np.arange(16.0).reshape(4, 4), aspect="auto")
        ax.plot([0, 1], [0, 1], label="a")
        ax.plot([1, 2], [1, 2], label="b")
        legend = jr.smart_legend(ax)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert legend._ncols == 2
        assert legend.get_window_extent(renderer).y0 >= ax.get_window_extent(renderer).y1

    def test_hline_label_at_a_free_end(self, dejavu):
        fig, ax = jr.journal_figure("acs")
        ax.plot([0, 1], [0, 0.1])
        line = ax.axhline(0.5)
        ax.set_ylim(0, 1)
        text = jr.label_hline(ax, line, "Reference")
        assert isinstance(text, Text)
        assert text.get_text() == "Reference"
        assert text.xy[0] == 1.0
        assert text.get_va() == "bottom"

    def test_hline_label_moves_off_data(self, dejavu):
        fig, ax = jr.journal_figure("acs")
        ax.plot([0.5, 1.0], [0.5, 0.5], lw=0, marker="s", ms=30)
        line = ax.axhline(0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        text = jr.label_hline(ax, line, "Reference")
        assert isinstance(text, Text)
        assert text.xy[0] == 0.0

    def test_hline_label_in_legend_when_blocked(self, dejavu):
        fig, ax = jr.journal_figure("acs")
        ax.imshow(np.arange(16.0).reshape(4, 4), aspect="auto", extent=(0, 1, 0, 1))
        line = ax.axhline(0.5)
        out = jr.label_hline(ax, line, "Reference")
        assert isinstance(out, Legend)
        assert [t.get_text() for t in out.get_texts()] == ["Reference"]

    def test_later_legends_keep_earlier_entries(self, dejavu):
        fig, ax = jr.journal_figure("acs")
        ax.imshow(np.arange(16.0).reshape(4, 4), aspect="auto", extent=(0, 1, 0, 1))
        jr.label_hline(ax, ax.axhline(0.5), "Reference")
        ax.plot([0, 1], [0.2, 0.8], label="line")
        legend = jr.smart_legend(ax)
        assert [t.get_text() for t in legend.get_texts()] == ["line", "Reference"]
        assert ax.get_legend() is legend


class TestAxes:
    @pytest.mark.parametrize(
        ("top", "power"), [(0.05, -2), (0.5, 0), (500.0, 0), (5000.0, 3)]
    )
    def test_sci_axis_si_range(self, dejavu, top, power):
        fig, ax = jr.journal_figure("acs")
        ax.plot([0, 1], [0, top])
        jr.sci_axis(ax)
        fig.canvas.draw()
        assert ax.yaxis.get_major_formatter().orderOfMagnitude == power

    def test_relative_error(self):
        eps = np.finfo(float).eps
        np.testing.assert_array_equal(jr.relative_error([2.0, 2.0], 2.0), [eps, eps])
        assert jr.relative_error(1.1, -1.0) == pytest.approx(2.1)
        assert jr.relative_error(1.1, 1.0) == pytest.approx(0.1)

    def test_log_error_axis(self, dejavu):
        fig, ax = jr.journal_figure("acs")
        ax.plot([0, 1, 2], [1e-3, 1e-8, 0.0])
        jr.log_error_axis(ax)
        assert ax.get_yscale() == "log"
        assert isinstance(ax.yaxis.get_major_formatter(), LogFormatterSciNotation)
        assert isinstance(ax.yaxis.get_minor_formatter(), NullFormatter)
        jr.log_error_axis(ax, "x")
        assert ax.get_xscale() == "log"
