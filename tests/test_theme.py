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
