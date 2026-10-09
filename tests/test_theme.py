"""The theme: every colour of the picture comes from Theme, so that a theme can change all of them."""

import ast
import pathlib
import re

import slopewalk

SOURCE = pathlib.Path(slopewalk.__file__).parent
COLOUR_KEYWORDS = {
    "color",
    "c",
    "fc",
    "ec",
    "mfc",
    "mec",
    "edgecolor",
    "facecolor",
    "colors",
    "markerfacecolor",
    "markeredgecolor",
    "foreground",
}
NAMED = {"k", "w", "r", "g", "b", "white", "black", "green", "red", "blue", "gray", "grey"}
COLORMAPS = {"Greys", "Greys_r", "RdBu", "RdBu_r", "bwr", "viridis", "coolwarm"}


def is_colour(value):
    if not isinstance(value, str):
        return False
    return bool(
        re.fullmatch(r"#[0-9A-Fa-f]{3}([0-9A-Fa-f]{3})?", value)
        or re.fullmatch(r"0?\.\d+|1\.0", value)
        or value in NAMED
        or value.startswith("tab:")
    )


def literals(name):
    """Colour and colour-map literals outside the Theme class of one module: (line, value)."""
    tree = ast.parse((SOURCE / name).read_text())
    theme = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.endswith("Theme"):
            theme |= set(range(node.lineno, node.end_lineno + 1))
    found = []
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) in theme:
            continue
        if isinstance(node, ast.keyword) and node.arg in COLOUR_KEYWORDS:
            found += [
                (node.lineno, v.value)
                for v in ast.walk(node.value)
                if isinstance(v, ast.Constant) and is_colour(v.value)
            ]
        if isinstance(node, ast.keyword) and node.arg == "cmap":
            found += [
                (node.lineno, v.value)
                for v in ast.walk(node.value)
                if isinstance(v, ast.Constant) and v.value in COLORMAPS
            ]
    return found


def test_no_colour_outside_the_theme():
    for name in ("draw.py", "views.py", "backprop.py", "picture.py", "static.py"):
        assert literals(name) == [], name


def luminance(colour):
    from matplotlib.colors import to_rgb

    def channel(v):
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in to_rgb(colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([luminance(a), luminance(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def check_palette(T):
    """Text at 4.5:1 on every surface it sits on; lines and marks that carry information at 3:1 on paper."""
    text_on = [
        (T.ink, T.paper),
        (T.muted, T.paper),
        (T.note_text, T.paper),
        (T.derivation_text, T.paper),
        (T.class_colour[0], T.paper),
        (T.class_colour[1], T.paper),
        (T.negative_text, T.paper),
        (T.backprop_colour, T.paper),
        (T.backprop_colour, T.backprop_tint),
        (T.frozen_text, T.frozen_fill),
        (T.ink, T.class_tint[0]),
        (T.ink, T.class_tint[1]),
        (T.ink, T.mixed_tint),
        (T.ink, T.neutral_tint),
        (T.ink, T.value_box["fc"]),
        (T.ink, T.table_head),
        (T.ink, T.table_highlight),
        (T.mixed_colour, T.paper),
        (T.neutral_colour, T.paper),
    ]
    lines = [
        T.positive,
        T.negative,
        T.hidden,
        T.gd_colour,
        T.momentum_colour,
        T.boundary,
        T.path_colour,
        T.prediction_colour,
        T.start_colour,
        T.class_colour[0],
        T.class_colour[1],
    ]
    low = [(a, b, round(contrast(a, b), 2)) for a, b in text_on if contrast(a, b) < 4.5]
    low += [(a, T.paper, round(contrast(a, T.paper), 2)) for a in lines if contrast(a, T.paper) < 3.0]
    assert low == []


def test_light_palette_contrast():
    from slopewalk.draw import THEME

    check_palette(THEME)
