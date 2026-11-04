# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Ruhi palette, colour science and perceptually ordered colour maps.

The Ruhi palette (Ruhila Goswami) has five colours. This module holds the
colour science used to check it and to build colour maps from it.

- sRGB primaries, white point and transfer function: IEC 61966-2-1:1999.
- CIELAB: ISO/CIE 11664-4:2019, D65 white.
- CIEDE2000 colour difference: ISO/CIE 11664-6:2014, as written out by
  Sharma, Wu and Dalal, Color Res. Appl. 30, 21 (2005).
- Colour-vision deficiency: Machado, Oliveira and Fernandes, IEEE Trans.
  Vis. Comput. Graph. 15, 1291 (2009), severity 1, applied to linear sRGB.
  Their tritan model is an approximation, not a model of tritanopia.
- Contrast ratio and thresholds: WCAG 2.1, success criteria 1.4.3 (text)
  and 1.4.11 (graphical objects).

Formula coefficients of these standards stay inline in the functions that
implement them.

```{versionadded} 1.14.0
```
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap, to_hex, to_rgb

if TYPE_CHECKING:
    from matplotlib.colors import Colormap
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike

    # One colour or several, in any form as_rgb takes.
    ColorLike = ColorType | Sequence[ColorType] | ArrayLike

RUHI_COLORS: dict[str, str] = {
    "coral": "#FF655D",
    "sunshine": "#F1DB4B",
    "teal": "#004D40",
    "sky": "#1E88E5",
    "magenta": "#D81B60",
}

# CIE 1931 xy chromaticities of the sRGB primaries and of D65.
_SRGB_PRIMARIES_XY = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
_D65_XY = (0.3127, 0.3290)
# sRGB transfer function: linear segment slope, offset, exponent, breakpoint.
_SRGB_SLOPE = 12.92
_SRGB_OFFSET = 0.055
_SRGB_GAMMA = 2.4
_SRGB_BREAK = 0.04045

# WCAG 2.1 minimum contrast ratios.
WCAG_TEXT_CONTRAST = 4.5
WCAG_GRAPHICS_CONTRAST = 3.0

# Machado et al. (2009), Table 1 at severity 1.0, for linear RGB.
_MACHADO_2009 = {
    "protan": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "deutan": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    "tritan": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ]
    ),
}
VISION_TYPES = ("normal", "protan", "deutan", "tritan")

# Bisection on chroma stops at float64 resolution.
_BISECTION_STEPS = np.finfo(float).nmant + 1
# Hex colours and raster output carry 8 bits per channel. Colour-map tables
# are refined until no entry moves by half a code value.
_CODE_MAX = np.iinfo(np.uint8).max
_TABLE_TOL = 0.5 / _CODE_MAX
# Hue angles are in degrees.
_HALF_TURN = 180.0
_FULL_TURN = 360.0


def _xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def _rgb_to_xyz_matrix() -> np.ndarray:
    primaries = np.column_stack([_xy_to_xyz(*xy) for xy in _SRGB_PRIMARIES_XY])
    scale = np.linalg.solve(primaries, _xy_to_xyz(*_D65_XY))
    return primaries * scale


_RGB_TO_XYZ = _rgb_to_xyz_matrix()
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_WHITE_XYZ = _xy_to_xyz(*_D65_XY)


def as_rgb(colors: ColorLike) -> np.ndarray:
    """Return colours (names, hex strings, RGB or RGBA) as an (..., 3) RGB array.

    ```{versionadded} 1.14.0
    ```
    """
    if isinstance(colors, str):
        return np.asarray(to_rgb(colors), dtype=float)
    arr = np.asarray(colors, dtype=object if _has_strings(colors) else float)
    if arr.dtype == object:
        return np.array([to_rgb(c) for c in colors], dtype=float)
    return arr[..., :3].astype(float)


def _has_strings(colors) -> bool:
    return isinstance(colors, Sequence) and any(isinstance(c, str) for c in colors)


def srgb_to_linear(rgb: ArrayLike) -> np.ndarray:
    """Decode gamma-encoded sRGB in [0, 1] to linear light.

    ```{versionadded} 1.14.0
    ```
    """
    rgb = np.asarray(rgb, dtype=float)
    low = rgb / _SRGB_SLOPE
    high = ((rgb + _SRGB_OFFSET) / (1.0 + _SRGB_OFFSET)) ** _SRGB_GAMMA
    return np.where(rgb <= _SRGB_BREAK, low, high)


def linear_to_srgb(lin: ArrayLike) -> np.ndarray:
    """Encode linear light in [0, 1] as gamma-encoded sRGB.

    ```{versionadded} 1.14.0
    ```
    """
    lin = np.asarray(lin, dtype=float)
    low = lin * _SRGB_SLOPE
    high = (1.0 + _SRGB_OFFSET) * np.abs(lin) ** (1.0 / _SRGB_GAMMA) - _SRGB_OFFSET
    return np.where(lin <= _SRGB_BREAK / _SRGB_SLOPE, low, high)


def _lab_f(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta**3, np.cbrt(t), t / (3.0 * delta**2) + 4.0 / 29.0)


def _lab_f_inv(f: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(f > delta, f**3, 3.0 * delta**2 * (f - 4.0 / 29.0))


def linear_to_lab(lin: ArrayLike) -> np.ndarray:
    """Convert linear sRGB to CIELAB (D65).

    ```{versionadded} 1.14.0
    ```
    """
    xyz = np.asarray(lin, dtype=float) @ _RGB_TO_XYZ.T
    fx, fy, fz = np.moveaxis(_lab_f(xyz / _WHITE_XYZ), -1, 0)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def lab_to_linear(lab: ArrayLike) -> np.ndarray:
    """Convert CIELAB (D65) to linear sRGB, without clipping.

    ```{versionadded} 1.14.0
    ```
    """
    l_star, a, b = np.moveaxis(np.asarray(lab, dtype=float), -1, 0)
    fy = (l_star + 16.0) / 116.0
    f = np.stack([fy + a / 500.0, fy, fy - b / 200.0], axis=-1)
    return (_lab_f_inv(f) * _WHITE_XYZ) @ _XYZ_TO_RGB.T


def srgb_to_lab(colors: ColorLike) -> np.ndarray:
    """Convert sRGB colours to CIELAB (D65).

    ```{versionadded} 1.14.0
    ```
    """
    return linear_to_lab(srgb_to_linear(as_rgb(colors)))


def lab_to_lch(lab: ArrayLike) -> np.ndarray:
    """Convert CIELAB to cylindrical CIELCh with hue in degrees.

    ```{versionadded} 1.14.0
    ```
    """
    l_star, a, b = np.moveaxis(np.asarray(lab, dtype=float), -1, 0)
    hue = np.degrees(np.arctan2(b, a)) % _FULL_TURN
    return np.stack([l_star, np.hypot(a, b), hue], axis=-1)


def lch_to_lab(lch: ArrayLike) -> np.ndarray:
    """Convert CIELCh (hue in degrees) to CIELAB.

    ```{versionadded} 1.14.0
    ```
    """
    l_star, chroma, hue = np.moveaxis(np.asarray(lch, dtype=float), -1, 0)
    rad = np.radians(hue)
    return np.stack([l_star, chroma * np.cos(rad), chroma * np.sin(rad)], axis=-1)


def srgb_to_lch(colors: ColorLike) -> np.ndarray:
    """Convert sRGB colours to CIELCh.

    ```{versionadded} 1.14.0
    ```
    """
    return lab_to_lch(srgb_to_lab(colors))


def _in_gamut(lin: np.ndarray) -> np.ndarray:
    return np.all((lin >= 0.0) & (lin <= 1.0), axis=-1)


def lch_to_srgb(lch: ArrayLike) -> np.ndarray:
    """Convert CIELCh to sRGB, reducing chroma at fixed L* and hue if needed.

    Out-of-gamut colours keep their lightness and hue; chroma is bisected
    down to the sRGB gamut boundary.

    ```{versionadded} 1.14.0
    ```
    """
    lch = np.atleast_2d(np.asarray(lch, dtype=float)).copy()
    lin = lab_to_linear(lch_to_lab(lch))
    out = ~_in_gamut(lin)
    if np.any(out):
        lo = np.zeros(out.sum())
        hi = lch[out, 1].copy()
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            trial = np.column_stack([lch[out, 0], mid, lch[out, 2]])
            ok = _in_gamut(lab_to_linear(lch_to_lab(trial)))
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        lch[out, 1] = lo
        lin = lab_to_linear(lch_to_lab(lch))
    return linear_to_srgb(np.clip(lin, 0.0, 1.0))


def delta_e_2000(lab1: ArrayLike, lab2: ArrayLike) -> np.ndarray:
    """CIEDE2000 colour difference with k_L = k_C = k_H = 1.

    ```{versionadded} 1.14.0
    ```
    """
    l1, a1, b1 = np.moveaxis(np.asarray(lab1, dtype=float), -1, 0)
    l2, a2, b2 = np.moveaxis(np.asarray(lab2, dtype=float), -1, 0)
    c_bar7 = ((np.hypot(a1, b1) + np.hypot(a2, b2)) / 2.0) ** 7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % _FULL_TURN
    h2p = np.degrees(np.arctan2(b2, a2p)) % _FULL_TURN
    chroma_zero = (c1p * c2p) == 0.0

    dh = h2p - h1p
    dh = np.where(
        dh > _HALF_TURN,
        dh - _FULL_TURN,
        np.where(dh < -_HALF_TURN, dh + _FULL_TURN, dh),
    )
    dh = np.where(chroma_zero, 0.0, dh)
    d_l = l2 - l1
    d_c = c2p - c1p
    d_h = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2.0))

    l_bar = (l1 + l2) / 2.0
    c_bar = (c1p + c2p) / 2.0
    h_sum = h1p + h2p
    h_bar = np.where(
        np.abs(h1p - h2p) <= _HALF_TURN,
        h_sum / 2.0,
        np.where(
            h_sum < _FULL_TURN, (h_sum + _FULL_TURN) / 2.0, (h_sum - _FULL_TURN) / 2.0
        ),
    )
    h_bar = np.where(chroma_zero, h_sum, h_bar)
    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar))
        + 0.32 * np.cos(np.radians(3.0 * h_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((h_bar - 275.0) / 25.0) ** 2))
    r_c = 2.0 * np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7))
    s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_bar
    s_h = 1.0 + 0.015 * c_bar * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c
    tl, tc, th = d_l / s_l, d_c / s_c, d_h / s_h
    return np.sqrt(tl**2 + tc**2 + th**2 + r_t * tc * th)


def simulate_cvd(colors: ColorLike, kind: str) -> np.ndarray:
    """Simulate how sRGB colours look with a colour-vision deficiency.

    Parameters
    ----------
    colors
        Colour names, hex strings or sRGB triples.
    kind
        ``"normal"``, ``"protan"``, ``"deutan"`` or ``"tritan"``.

    ```{versionadded} 1.14.0
    ```
    """
    rgb = as_rgb(colors)
    if kind == "normal":
        return rgb
    try:
        matrix = _MACHADO_2009[kind]
    except KeyError:
        msg = f"Unknown vision type {kind!r}; expected one of {VISION_TYPES}"
        raise ValueError(msg) from None
    lin = srgb_to_linear(rgb) @ matrix.T
    return linear_to_srgb(np.clip(lin, 0.0, 1.0))


def relative_luminance(colors: ColorLike) -> np.ndarray:
    """WCAG relative luminance (the Y of linear sRGB).

    ```{versionadded} 1.14.0
    ```
    """
    return srgb_to_linear(as_rgb(colors)) @ _RGB_TO_XYZ[1]


def contrast_ratio(color: ColorLike, background: ColorLike = "white") -> float:
    """WCAG 2.1 contrast ratio between two colours, from 1 to 21.

    ```{versionadded} 1.14.0
    ```
    """
    y1 = float(relative_luminance(color))
    y2 = float(relative_luminance(background))
    hi, lo = max(y1, y2), min(y1, y2)
    return (hi + 0.05) / (lo + 0.05)


def readable_color(
    color: ColorLike, background: ColorLike = "white", ratio: float = WCAG_TEXT_CONTRAST
) -> str:
    """Return ``color``, or its shade at the same hue that reaches ``ratio``.

    The shade keeps hue, scales chroma with lightness and moves L* away from
    the background until the WCAG contrast ratio is met. The ratio holds for
    the returned 8-bit hex value, not only before rounding. Use it for direct
    labels (ratio 4.5, text) and outlines (ratio 3, graphics).

    ```{versionadded} 1.14.0
    ```
    """
    rgb = np.round(as_rgb(color) * _CODE_MAX) / _CODE_MAX
    if contrast_ratio(rgb, background) >= ratio:
        return to_hex(rgb)
    l_star, chroma, hue = srgb_to_lch(rgb)
    darken = contrast_ratio("black", background) >= contrast_ratio("white", background)

    def shade(frac: float) -> np.ndarray:
        if darken:
            lch = [l_star * (1.0 - frac), chroma * (1.0 - frac), hue]
        else:
            lch = [l_star + (100.0 - l_star) * frac, chroma, hue]
        return np.round(lch_to_srgb(lch)[0] * _CODE_MAX) / _CODE_MAX

    lo, hi = 0.0, 1.0
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if contrast_ratio(shade(mid), background) >= ratio:
            hi = mid
        else:
            lo = mid
    return to_hex(shade(hi))


def best_contrast(
    neighbours: ColorLike, candidates: Sequence[str] = ("black", "white")
) -> str:
    """The candidate whose smallest WCAG contrast with ``neighbours`` is largest.

    Picks a line colour that stays visible against every colour it borders,
    such as a contour line between two filled bands.

    ```{versionadded} 1.14.0
    ```
    """
    neighbours = as_rgb(neighbours).reshape(-1, 3)
    return max(candidates, key=lambda c: min(contrast_ratio(c, n) for n in neighbours))


def contrast_limited(
    cmap: str | Colormap,
    background: ColorLike = "white",
    ratio: float = WCAG_GRAPHICS_CONTRAST,
) -> ListedColormap:
    """The entries of ``cmap`` that reach ``ratio`` against ``background``.

    Lines coloured from the result all meet the WCAG 1.4.11 contrast for
    graphics. For maps with monotone lightness the entries kept are one end
    of the map, in their original order.

    ```{versionadded} 1.14.0
    ```
    """
    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    rgb = cmap(np.linspace(0.0, 1.0, cmap.N))[:, :3]
    y_bg = float(relative_luminance(background))
    y = relative_luminance(rgb)
    ratios = (np.maximum(y, y_bg) + 0.05) / (np.minimum(y, y_bg) + 0.05)
    keep = ratios >= ratio
    if not keep.any():
        msg = f"No entry of {cmap.name} reaches a contrast of {ratio} on {background}"
        raise ValueError(msg)
    return ListedColormap(rgb[keep], name=f"{cmap.name}_contrast")


def pairwise_delta_e(colors: ColorLike, kind: str = "normal") -> np.ndarray:
    """Matrix of CIEDE2000 differences between colours seen with ``kind``.

    ```{versionadded} 1.14.0
    ```
    """
    lab = linear_to_lab(srgb_to_linear(simulate_cvd(colors, kind)))
    return delta_e_2000(lab[:, None, :], lab[None, :, :])


def min_pairwise_delta_e(colors: ColorLike, kind: str = "normal") -> float:
    """Smallest CIEDE2000 difference between any two of ``colors``.

    ```{versionadded} 1.14.0
    ```
    """
    d = pairwise_delta_e(colors, kind)
    return float(d[np.triu_indices(len(d), k=1)].min())


# Okabe and Ito (2008), "Color Universal Design", the usual reference
# palette for readers with colour-vision deficiencies (Wong, Nat. Methods
# 8, 441 (2011)).
OKABE_ITO = (
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
)


def order_for_lines(
    colors: Mapping[str, str], background: ColorLike = "white"
) -> list[str]:
    """Order named colours for line plots on ``background``.

    Colours that pass the WCAG 1.4.11 graphics contrast go first, the most
    contrasting one leading. Each next colour is the one whose smallest
    CIEDE2000 distance to those already chosen, over normal vision and the
    three simulated deficiencies, is largest. Colours that fail the contrast
    go last, ordered the same way.

    ```{versionadded} 1.14.0
    ```
    """
    names = list(colors)
    passing = [
        n
        for n in names
        if contrast_ratio(colors[n], background) >= WCAG_GRAPHICS_CONTRAST
    ]
    failing = [n for n in names if n not in passing]
    chosen: list[str] = []
    for group in (passing, failing):
        pool = sorted(group, key=lambda n: -contrast_ratio(colors[n], background))
        while pool:
            if not chosen:
                pick = pool[0]
            else:
                pick = max(
                    pool,
                    key=lambda n: _cvd_distance(colors[n], [colors[c] for c in chosen]),
                )
            chosen.append(pick)
            pool.remove(pick)
    return chosen


def _cvd_distance(color, others: Iterable) -> float:
    others = list(others)
    best = np.inf
    for kind in VISION_TYPES:
        lab = linear_to_lab(srgb_to_linear(simulate_cvd([color, *others], kind)))
        best = min(best, float(delta_e_2000(lab[0], lab[1:]).min()))
    return best


RUHI_LINE_ORDER = tuple(order_for_lines(RUHI_COLORS))
RUHI_CYCLE = tuple(RUHI_COLORS[name] for name in RUHI_LINE_ORDER)


# --- Perceptually ordered colour maps -------------------------------------


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson slopes for a monotone cubic Hermite interpolant."""
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros_like(y)
    if len(x) == 2:  # noqa: PLR2004
        m[:] = delta[0]
        return m
    for k in range(1, len(x) - 1):
        if delta[k - 1] * delta[k] > 0.0:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    m[0] = _pchip_end_slope(h[0], h[1], delta[0], delta[1])
    m[-1] = _pchip_end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return m


def _pchip_end_slope(h0, h1, d0, d1) -> float:
    m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(m) != np.sign(d0):
        return 0.0
    if np.sign(d0) != np.sign(d1) and abs(m) > abs(3.0 * d0):
        return 3.0 * d0
    return m


def _pchip(x, y, xq) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)
    m = _pchip_slopes(x, y)
    k = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    h = x[k + 1] - x[k]
    t = (xq - x[k]) / h
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)
    return h00 * y[k] + h10 * h * m[k] + h01 * y[k + 1] + h11 * h * m[k + 1]


def _lightness_path(anchors: np.ndarray):
    """Return ``f(L) -> (L, C, h)`` through LCh anchors sorted by lightness."""
    l_star, chroma, hue = anchors.T

    def path(lq):
        return np.column_stack([lq, _pchip(l_star, chroma, lq), _pchip(l_star, hue, lq)])

    return path


def _even_in_arc_length(paths, l0: float, l1: float, positions) -> np.ndarray:
    """Sample ``paths`` at shared lightness values set by arc-length ``positions``.

    ``positions`` run from 0 at ``l0`` to 1 at ``l1`` and are fractions of the
    CIEDE2000 arc length summed over all paths, so one path is spaced evenly
    and several paths share one lightness schedule. Returns an array of shape
    ``(len(paths), len(positions), 3)``.
    """
    positions = np.asarray(positions, dtype=float)
    tables = None
    samples = len(positions)
    while True:
        lq = np.linspace(l0, l1, samples)
        arc = np.zeros(samples)
        for path in paths:
            lab = linear_to_lab(srgb_to_linear(lch_to_srgb(path(lq))))
            arc[1:] += np.cumsum(delta_e_2000(lab[:-1], lab[1:]))
        lp = np.interp(positions * arc[-1], arc, lq)
        new = np.stack([lch_to_srgb(path(lp)) for path in paths])
        if tables is not None and np.max(np.abs(new - tables)) <= _TABLE_TOL:
            return new
        tables = new
        samples = 2 * samples - 1


def _unwrap_increasing(hues: Sequence[float]) -> np.ndarray:
    """Unwrap hues so that each is at least the one before it."""
    out = [float(hues[0])]
    for value in hues[1:]:
        hue = float(value)
        while hue < out[-1]:
            hue += _FULL_TURN
        out.append(hue)
    return np.array(out)


def _ruhi_lch() -> dict[str, np.ndarray]:
    return dict(zip(RUHI_COLORS, srgb_to_lch(list(RUHI_COLORS.values())), strict=True))


def _anchors(rows) -> np.ndarray:
    lch = np.array(rows, dtype=float)
    lch[:, 2] = _unwrap_increasing(lch[:, 2])
    return lch


def ruhi_sequential_colors(n: int | None = None) -> np.ndarray:
    """sRGB table of the sequential Ruhi map, dark wine to sunshine.

    The anchors are magenta, coral and sunshine, whose lightness and hue
    rise together, below a dark end with teal's lightness and chroma at
    magenta's hue, so the map spans the palette's lightness range. Chroma
    and hue follow monotone cubic curves in L* and the table is spaced
    evenly in CIEDE2000 arc length (Kovesi, arXiv:1509.03700). Teal and sky
    are left out. Sky's lightness lies between magenta's and coral's, so a
    map through it turns the hue by 94 and then 116 degrees over 15.5 units
    of L*, and its lightness is no longer monotone for simulated protan,
    deutan or tritan vision. Starting at teal itself turns the hue by 166
    degrees on the way to magenta, and lightness is no longer monotone for
    simulated protan vision.

    ```{versionadded} 1.14.0
    ```
    """
    n = n or mpl.rcParams["image.lut"]
    c = _ruhi_lch()
    dark = [c["teal"][0], c["teal"][1], c["magenta"][2]]
    lch = _anchors([dark, c["magenta"], c["coral"], c["sunshine"]])
    path = _lightness_path(lch)
    return _even_in_arc_length([path], lch[0, 0], lch[-1, 0], np.linspace(0.0, 1.0, n))[0]


def ruhi_diverging_colors(n: int | None = None) -> np.ndarray:
    """sRGB table of the diverging Ruhi map, teal through grey to dark wine.

    The cool arm runs from teal through sky, the warm arm from a dark end
    (teal's lightness and chroma at magenta's hue) through magenta and
    coral. Both meet in a neutral grey at sunshine's lightness. The arms
    share one lightness schedule, spaced evenly in their summed CIEDE2000
    arc length, so entries equally far from the centre have equal L*
    (Moreland, ISVC 2009).

    ```{versionadded} 1.14.0
    ```
    """
    n = n or mpl.rcParams["image.lut"]
    c = _ruhi_lch()
    l_end, l_mid = c["teal"][0], c["sunshine"][0]
    cool = _anchors([c["teal"], c["sky"], [l_mid, 0.0, c["sky"][2]]])
    dark = [l_end, c["teal"][1], c["magenta"][2]]
    warm = _anchors([dark, c["magenta"], c["coral"], [l_mid, 0.0, c["coral"][2]]])
    u = np.linspace(-1.0, 1.0, n)
    arms = [_lightness_path(cool), _lightness_path(warm)]
    cool_rgb, warm_rgb = _even_in_arc_length(arms, l_end, l_mid, 1.0 - np.abs(u))
    return np.where((u <= 0.0)[:, None], cool_rgb, warm_rgb)


RUHI_SEQUENTIAL = "ruhi_seq"
RUHI_DIVERGING = "ruhi_div"


def register_ruhi_colormaps() -> None:
    """Register ``ruhi_seq`` and ``ruhi_div`` (and their ``_r``) with matplotlib.

    ```{versionadded} 1.14.0
    ```
    """
    builders = {
        RUHI_SEQUENTIAL: ruhi_sequential_colors,
        RUHI_DIVERGING: ruhi_diverging_colors,
    }
    for name, build in builders.items():
        if name in mpl.colormaps:
            continue
        cmap = ListedColormap(build(), name=name)
        mpl.colormaps.register(cmap)
        mpl.colormaps.register(cmap.reversed(), name=f"{name}_r")


@dataclass(frozen=True)
class ColormapReport:
    """Perceptual checks of a colour map, by vision type.

    ```{versionadded} 1.14.0
    ```
    """

    name: str
    lightness: dict[str, np.ndarray]
    steps: dict[str, np.ndarray]

    def lightness_monotonic(self, kind: str = "normal") -> bool:
        """True if L* rises or falls strictly along the whole map."""
        d = np.diff(self.lightness[kind])
        return bool(np.all(d > 0.0) or np.all(d < 0.0))

    def arms_monotonic(self, kind: str = "normal") -> bool:
        """True if L* is strictly monotone on each side of the centre, in
        opposite directions, as a diverging map needs."""
        l_star = self.lightness[kind]
        mid = len(l_star) // 2
        left, right = np.diff(l_star[: mid + len(l_star) % 2]), np.diff(l_star[mid:])
        rising = np.all(left > 0.0) and np.all(right < 0.0)
        falling = np.all(left < 0.0) and np.all(right > 0.0)
        return bool(rising or falling)

    def step_cv(self, kind: str = "normal") -> float:
        """Coefficient of variation of CIEDE2000 steps between entries."""
        s = self.steps[kind]
        return float(s.std() / s.mean())


def colormap_report(cmap: str | Colormap) -> ColormapReport:
    """Lightness and CIEDE2000 step profile of a colour map for each vision type.

    ```{versionadded} 1.14.0
    ```
    """
    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    rgb = cmap(np.linspace(0.0, 1.0, cmap.N))[:, :3]
    lightness, steps = {}, {}
    for kind in VISION_TYPES:
        lab = linear_to_lab(srgb_to_linear(simulate_cvd(rgb, kind)))
        lightness[kind] = lab[:, 0]
        steps[kind] = delta_e_2000(lab[:-1], lab[1:])
    return ColormapReport(cmap.name, lightness, steps)


@dataclass(frozen=True)
class PaletteReport:
    """WCAG contrast and CVD separation of a categorical palette.

    ```{versionadded} 1.14.0
    ```
    """

    names: tuple[str, ...]
    hex: tuple[str, ...]
    lch: np.ndarray
    contrast: tuple[float, ...]
    min_delta_e: dict[str, float]
    reference_min_delta_e: dict[str, float]

    def passes_graphics_contrast(self) -> tuple[str, ...]:
        """Names of the colours that reach the WCAG 1.4.11 contrast for graphics."""
        return tuple(
            n
            for n, c in zip(self.names, self.contrast, strict=True)
            if c >= WCAG_GRAPHICS_CONTRAST
        )

    def at_least_reference(self, kind: str) -> bool:
        """True if the palette separates its colours seen with ``kind`` at least as
        well as the reference palette does (smallest CIEDE2000 difference)."""
        return self.min_delta_e[kind] >= self.reference_min_delta_e[kind]


def palette_report(
    colors: Mapping[str, str] | None = None,
    background: ColorLike = "white",
    reference: Sequence[str] = OKABE_ITO,
) -> PaletteReport:
    """Check a categorical palette against WCAG and simulated CVD.

    ``min_delta_e[kind]`` is the smallest CIEDE2000 difference between any two
    colours seen with ``kind``; ``reference_min_delta_e`` is the same number
    for ``reference`` (Okabe-Ito by default).

    ```{versionadded} 1.14.0
    ```
    """
    colors = dict(colors or RUHI_COLORS)
    names = tuple(colors)
    values = [colors[n] for n in names]
    return PaletteReport(
        names=names,
        hex=tuple(to_hex(as_rgb(v)) for v in values),
        lch=srgb_to_lch(values),
        contrast=tuple(contrast_ratio(v, background) for v in values),
        min_delta_e={k: min_pairwise_delta_e(values, k) for k in VISION_TYPES},
        reference_min_delta_e={
            k: min_pairwise_delta_e(reference, k) for k in VISION_TYPES
        },
    )


register_ruhi_colormaps()
