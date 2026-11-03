# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Colour science, Ruhi palette checks and the Ruhi colour maps."""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import ListedColormap

from chemparseplot.plot import palette as pal

# Agreement to half the digits of double precision, relative to the range of
# the quantity: L* runs over 100, sRGB channels over 1.
ROUND_OFF = math.sqrt(np.finfo(float).eps)
L_STAR_RANGE = 100.0

# Sharma, Wu and Dalal, Color Res. Appl. 30, 21 (2005), Table 1: the 34 test
# pairs for CIEDE2000 (L1, a1, b1, L2, a2, b2, Delta E00). The values are
# given to four decimals, so a correct implementation agrees to half a unit
# in the fourth.
SHARMA_2005 = (
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461),
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    (50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977),
    (50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030),
    (50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731),
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082),
)
SHARMA_TOL = 0.5e-4

# Machado et al. (2009) print their matrices to six decimals, so each row of
# three sums to one within three half-units of the sixth decimal.
MACHADO_ROW_TOL = 3 * 0.5e-6

RUHI_HEX = list(pal.RUHI_COLORS.values())
CVD_TYPES = [k for k in pal.VISION_TYPES if k != "normal"]


@pytest.fixture(scope="module")
def sharma():
    table = np.array(SHARMA_2005)
    return table[:, :3], table[:, 3:6], table[:, 6]


class TestCIEDE2000:
    def test_sharma_table(self, sharma):
        lab1, lab2, expected = sharma
        np.testing.assert_allclose(
            pal.delta_e_2000(lab1, lab2), expected, rtol=0, atol=SHARMA_TOL
        )

    def test_symmetric(self, sharma):
        # Exact: every term is symmetric in the two colours or only changes
        # sign, which IEEE arithmetic does without rounding.
        lab1, lab2, _ = sharma
        np.testing.assert_array_equal(
            pal.delta_e_2000(lab1, lab2), pal.delta_e_2000(lab2, lab1)
        )

    def test_zero_for_identical_colours(self, sharma):
        lab1, _, _ = sharma
        np.testing.assert_array_equal(pal.delta_e_2000(lab1, lab1), 0.0)


class TestConversions:
    def test_white_and_black(self):
        np.testing.assert_allclose(
            pal.srgb_to_lab("white"), [100.0, 0.0, 0.0], atol=ROUND_OFF * L_STAR_RANGE
        )
        np.testing.assert_allclose(
            pal.srgb_to_lab("black"), [0.0, 0.0, 0.0], atol=ROUND_OFF * L_STAR_RANGE
        )

    def test_greys_are_neutral_and_ordered(self):
        levels = np.arange(pal._CODE_MAX + 1) / pal._CODE_MAX
        lab = pal.srgb_to_lab(np.repeat(levels[:, None], 3, axis=1))
        np.testing.assert_allclose(lab[:, 1:], 0.0, atol=ROUND_OFF * L_STAR_RANGE)
        assert np.all(np.diff(lab[:, 0]) > 0.0)

    def test_srgb_lab_round_trip(self):
        rng = np.random.default_rng(0)
        rgb = np.vstack([pal.as_rgb(RUHI_HEX), rng.random((256, 3))])
        lab = pal.srgb_to_lab(rgb)
        back = pal.linear_to_srgb(pal.lab_to_linear(lab))
        np.testing.assert_allclose(back, rgb, atol=ROUND_OFF)

    def test_lch_round_trip(self):
        lab = pal.srgb_to_lab(RUHI_HEX)
        np.testing.assert_allclose(
            pal.lch_to_lab(pal.lab_to_lch(lab)), lab, atol=ROUND_OFF * L_STAR_RANGE
        )

    def test_as_rgb_accepts_names_hex_and_rgba(self):
        np.testing.assert_array_equal(pal.as_rgb("white"), [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(
            pal.as_rgb(["#000000", "white"]), [[0, 0, 0], [1, 1, 1]]
        )
        np.testing.assert_array_equal(
            pal.as_rgb([[0.1, 0.2, 0.3, 0.5]]), [[0.1, 0.2, 0.3]]
        )

    def test_gamut_mapping_keeps_lightness_and_hue(self):
        lch = np.array([[50.0, 200.0, 30.0], [80.0, 150.0, 250.0], [20.0, 90.0, 140.0]])
        rgb = pal.lch_to_srgb(lch)
        assert np.all((rgb >= 0.0) & (rgb <= 1.0))
        out = pal.srgb_to_lch(rgb)
        np.testing.assert_allclose(out[:, 0], lch[:, 0], atol=ROUND_OFF * L_STAR_RANGE)
        np.testing.assert_allclose(out[:, 2], lch[:, 2], atol=ROUND_OFF * 360.0)
        assert np.all(out[:, 1] < lch[:, 1])

    def test_in_gamut_colours_pass_unchanged(self):
        lch = pal.srgb_to_lch(RUHI_HEX)
        np.testing.assert_allclose(
            pal.lch_to_srgb(lch), pal.as_rgb(RUHI_HEX), atol=ROUND_OFF
        )


class TestColourVisionDeficiency:
    @pytest.mark.parametrize("kind", CVD_TYPES)
    def test_machado_rows_sum_to_one(self, kind):
        np.testing.assert_allclose(
            pal._MACHADO_2009[kind].sum(axis=1), 1.0, rtol=0, atol=MACHADO_ROW_TOL
        )

    @pytest.mark.parametrize("kind", pal.VISION_TYPES)
    def test_white_and_black_are_kept(self, kind):
        np.testing.assert_allclose(
            pal.simulate_cvd("white", kind), 1.0, rtol=0, atol=MACHADO_ROW_TOL
        )
        np.testing.assert_array_equal(pal.simulate_cvd("black", kind), 0.0)

    def test_normal_is_identity(self):
        np.testing.assert_array_equal(
            pal.simulate_cvd(RUHI_HEX, "normal"), pal.as_rgb(RUHI_HEX)
        )

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown vision type"):
            pal.simulate_cvd(RUHI_HEX, "achromat")


class TestContrast:
    def test_black_on_white_is_the_wcag_maximum(self):
        assert pal.contrast_ratio("black", "white") == pytest.approx(21.0, rel=ROUND_OFF)
        assert pal.contrast_ratio("white", "black") == pal.contrast_ratio(
            "black", "white"
        )
        assert pal.contrast_ratio("#1E88E5", "#1E88E5") == 1.0

    def test_ruhi_graphics_contrast_on_white(self):
        report = pal.palette_report()
        assert report.passes_graphics_contrast() == ("teal", "sky", "magenta")

    @pytest.mark.parametrize("background", ["white", "black", "#F1DB4B"])
    @pytest.mark.parametrize(
        "ratio", [pal.WCAG_GRAPHICS_CONTRAST, pal.WCAG_TEXT_CONTRAST]
    )
    def test_readable_color_meets_the_ratio_after_rounding(self, background, ratio):
        for color in RUHI_HEX:
            shade = pal.readable_color(color, background, ratio)
            assert shade.startswith("#")
            assert len(shade) == len("#rrggbb")
            assert pal.contrast_ratio(shade, background) >= ratio

    def test_readable_color_keeps_passing_colours(self):
        assert pal.readable_color("#004D40", "white") == "#004d40"

    def test_best_contrast(self):
        assert pal.best_contrast(["white", "#F1DB4B"]) == "black"
        assert pal.best_contrast(["black", "#004D40"]) == "white"

    def test_contrast_limited_keeps_the_dark_end_of_ruhi_seq(self):
        full = mpl.colormaps[pal.RUHI_SEQUENTIAL]
        kept = pal.contrast_limited(pal.RUHI_SEQUENTIAL)
        rgb = kept(np.arange(kept.N))[:, :3]
        assert all(
            pal.contrast_ratio(c, "white") >= pal.WCAG_GRAPHICS_CONTRAST for c in rgb
        )
        np.testing.assert_array_equal(rgb, full(np.arange(kept.N))[:, :3])
        assert pal.contrast_ratio(full(kept.N)[:3], "white") < pal.WCAG_GRAPHICS_CONTRAST

    def test_contrast_limited_rejects_an_unreachable_ratio(self):
        with pytest.raises(ValueError, match="No entry"):
            pal.contrast_limited(pal.RUHI_SEQUENTIAL, ratio=22.0)


class TestCategorical:
    def test_report_against_okabe_ito(self):
        report = pal.palette_report()
        assert report.names == tuple(pal.RUHI_COLORS)
        assert set(report.min_delta_e) == set(pal.VISION_TYPES)
        for kind in pal.VISION_TYPES:
            assert report.reference_min_delta_e[kind] == pytest.approx(
                pal.min_pairwise_delta_e(pal.OKABE_ITO, kind)
            )
        # Measured, not designed: the closest Ruhi pair is closer than the
        # closest Okabe-Ito pair except for deutan vision.
        assert [report.at_least_reference(k) for k in pal.VISION_TYPES] == [
            False,
            False,
            True,
            False,
        ]

    @pytest.mark.parametrize(
        ("kind", "pair"),
        [("protan", {"teal", "magenta"}), ("tritan", {"coral", "magenta"})],
    )
    def test_closest_pairs_named_in_the_theme(self, kind, pair):
        names = list(pal.RUHI_COLORS)
        d = pal.pairwise_delta_e(RUHI_HEX, kind)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        assert {names[i], names[j]} == pair

    def test_line_order_puts_contrasting_colours_first(self):
        order = pal.RUHI_LINE_ORDER
        ratios = [pal.contrast_ratio(pal.RUHI_COLORS[n], "white") for n in order]
        passing = [r >= pal.WCAG_GRAPHICS_CONTRAST for r in ratios]
        assert passing == sorted(passing, reverse=True)
        assert ratios[0] == max(ratios)
        assert pal.RUHI_CYCLE == tuple(pal.RUHI_COLORS[n] for n in order)


class TestColormaps:
    def test_registered(self):
        for name in (pal.RUHI_SEQUENTIAL, pal.RUHI_DIVERGING):
            assert mpl.colormaps[name].N == mpl.rcParamsDefault["image.lut"]
            assert f"{name}_r" in mpl.colormaps
        pal.register_ruhi_colormaps()

    @pytest.mark.parametrize("kind", pal.VISION_TYPES)
    def test_sequential_lightness_is_monotone(self, kind):
        assert pal.colormap_report(pal.RUHI_SEQUENTIAL).lightness_monotonic(kind)

    def test_sequential_is_more_even_than_viridis(self):
        ruhi = pal.colormap_report(pal.RUHI_SEQUENTIAL).step_cv()
        assert ruhi < pal.colormap_report("viridis").step_cv()

    def test_sequential_runs_dark_to_light_over_the_palette(self):
        lightness = pal.colormap_report(pal.RUHI_SEQUENTIAL).lightness["normal"]
        lch = pal.srgb_to_lch(RUHI_HEX)
        names = list(pal.RUHI_COLORS)
        tol = ROUND_OFF * L_STAR_RANGE
        assert lightness[0] == pytest.approx(lch[names.index("teal"), 0], abs=tol)
        assert lightness[-1] == pytest.approx(lch[names.index("sunshine"), 0], abs=tol)

    @pytest.mark.parametrize("kind", pal.VISION_TYPES)
    def test_diverging_arms_are_monotone(self, kind):
        assert pal.colormap_report(pal.RUHI_DIVERGING).arms_monotonic(kind)

    def test_diverging_arms_share_lightness(self):
        lightness = pal.colormap_report(pal.RUHI_DIVERGING).lightness["normal"]
        np.testing.assert_allclose(
            lightness, lightness[::-1], rtol=0, atol=ROUND_OFF * L_STAR_RANGE
        )

    @staticmethod
    def _sequential_through(anchors, n):
        """Map built as ruhi_seq is, through LCh ``anchors``, hue by the shorter turn."""
        lch = np.array(sorted(anchors, key=lambda row: row[0]), dtype=float)
        lch[:, 2] = np.degrees(np.unwrap(np.radians(lch[:, 2])))
        path = pal._lightness_path(lch)
        positions = np.linspace(0.0, 1.0, n)
        table = pal._even_in_arc_length([path], lch[0, 0], lch[-1, 0], positions)[0]
        return ListedColormap(table)

    def test_teal_or_sky_would_break_monotone_lightness(self):
        n = mpl.rcParamsDefault["image.lut"]
        c = pal._ruhi_lch()
        dark = [c["teal"][0], c["teal"][1], c["magenta"][2]]
        warm = [c["magenta"], c["coral"], c["sunshine"]]
        shipped = self._sequential_through([dark, *warm], n)
        np.testing.assert_allclose(
            shipped.colors, pal.ruhi_sequential_colors(n), rtol=0, atol=ROUND_OFF
        )
        sky = pal.colormap_report(self._sequential_through([dark, c["sky"], *warm], n))
        teal = pal.colormap_report(self._sequential_through([c["teal"], *warm], n))
        assert sky.lightness_monotonic("normal")
        assert not any(sky.lightness_monotonic(kind) for kind in CVD_TYPES)
        assert teal.lightness_monotonic("normal")
        protan, deutan, tritan = (teal.lightness_monotonic(kind) for kind in CVD_TYPES)
        assert not protan
        assert deutan
        assert tritan

    def test_odd_tables(self):
        seq = pal.ruhi_sequential_colors(5)
        div = pal.ruhi_diverging_colors(5)
        assert seq.shape == div.shape == (5, 3)
        centre = pal.srgb_to_lch(div[2])
        assert centre[1] == pytest.approx(0.0, abs=ROUND_OFF * L_STAR_RANGE)

    def test_report_detects_non_monotone_maps(self):
        assert not pal.colormap_report("ruhi_diverging").lightness_monotonic()
        assert not pal.colormap_report(pal.RUHI_DIVERGING).lightness_monotonic()
        assert not pal.colormap_report(pal.RUHI_SEQUENTIAL).arms_monotonic()


class TestElementColors:
    @staticmethod
    def _body():
        return {k: v for k, v in pal.RUHI_COLORS.items() if k != "coral"}

    def test_anchors_keep_colours_they_already_have(self):
        body = self._body()
        reference = dict(zip(pal.CPK_ANCHORS, body.values(), strict=True))
        reference["S"] = reference["O"]
        shuffled = {name: body[name] for name in reversed(list(body))}
        table = pal.assign_element_colors(reference, shuffled)
        assert {a: table[a] for a in pal.CPK_ANCHORS} == {
            a: reference[a] for a in pal.CPK_ANCHORS
        }
        assert table["S"] == table["O"]

    def test_carbon_and_nitrogen_part_on_the_jmol_colours(self):
        data = pytest.importorskip("ase.data")
        jmol_colors = pytest.importorskip("ase.data.colors").jmol_colors
        jmol = dict(
            zip(data.chemical_symbols[1 : len(jmol_colors)], jmol_colors[1:], strict=True)
        )
        body = self._body()
        names = list(body)
        body_lab = pal.srgb_to_lab([body[n] for n in names])
        nearest = {
            a: names[int(pal.delta_e_2000(pal.srgb_to_lab(jmol[a]), body_lab).argmin())]
            for a in ("C", "N")
        }
        assert nearest == {"C": "sky", "N": "sky"}
        table = pal.assign_element_colors(jmol, body)
        assert {a: table[a] for a in pal.CPK_ANCHORS} == {
            "H": body["sunshine"],
            "C": body["teal"],
            "N": body["sky"],
            "O": body["magenta"],
        }
        assert len(table) == len(jmol_colors) - 1

    def test_one_colour_per_anchor(self):
        with pytest.raises(ValueError, match="pair one to one"):
            pal.assign_element_colors({"H": "white"}, {"a": "black"}, anchors=("H", "C"))
        with pytest.raises(KeyError, match="no colour for C"):
            pal.assign_element_colors(
                {"H": "white"}, {"a": "black", "b": "grey"}, anchors=("H", "C")
            )
