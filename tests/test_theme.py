"""The theme: every colour of the picture comes from Theme, so that a theme can change all of them."""

import ast
import pathlib
import re

import pytest
from matplotlib.colors import to_rgb

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
    for name in ("theme.py", "draw.py", "views.py", "backprop.py", "picture.py", "static.py"):
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
        *[(T.class_colour[k], T.paper) for k in T.class_colour],
        *[(T.ink, T.class_tint[k]) for k in T.class_tint],
        (T.negative_text, T.paper),
        (T.backprop_colour, T.paper),
        (T.backprop_colour, T.backprop_tint),
        (T.frozen_text, T.frozen_fill),
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
        *T.class_colour.values(),
    ]
    low = [(a, b, round(contrast(a, b), 2)) for a, b in text_on if contrast(a, b) < 4.5]
    low += [(a, T.paper, round(contrast(a, T.paper), 2)) for a in lines if contrast(a, T.paper) < 3.0]
    assert low == []


def test_light_palette_contrast():
    from slopewalk.draw import THEME

    check_palette(THEME)


def test_dark_palette_contrast():
    from slopewalk.draw import DARK_THEME

    check_palette(DARK_THEME)


def drawn_texts(fig):
    """The texts a draw shows: figure and axes texts, titles, legends, table cells, and the tick and axis labels of
    axes whose axis is on (the network panels turn theirs off)."""
    fig.canvas.draw()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax._left_title, ax._right_title]
        if ax.axison:
            for axis in [ax.xaxis, ax.yaxis] + ([ax.zaxis] if hasattr(ax, "zaxis") else []):
                texts += [axis.label] + list(axis.get_ticklabels())
        if ax.get_legend() is not None:
            texts += list(ax.get_legend().get_texts())
        for table in ax.tables:
            texts += [cell.get_text() for cell in table.get_celld().values()]
    return [t for t in texts if t.get_visible() and t.get_text().strip()]


def every_text_is_legible(fig, T):
    """After a draw, every text the figure shows has 3:1 or more against paper."""
    return [(t.get_text()[:30], t.get_color()) for t in drawn_texts(fig) if contrast(t.get_color(), T.paper) < 3.0]


def test_dark_pictures_have_no_dark_text(monkeypatch):
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from helpers import make, sine_data

    from slopewalk import clusters, explore, xor_centres
    from slopewalk.draw import DARK_THEME, THEME

    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    calls = [
        lambda t: explore(make("tanh3"), *xor_centres(noise=0.35), theme=t),
        lambda t: explore(
            make("tanh3"), *xor_centres(), views=("cases", "slice", "line", "hidden", "momentum"), theme=t
        ),
        lambda t: explore(make("tanh3"), *xor_centres(noise=0.35), views="backprop", backprop_step=5, theme=t),
        lambda t: explore(
            nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 1)), *sine_data(), task="regression", theme=t
        ),
        lambda t: explore(
            nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 2)),
            torch.linspace(-2, 2, 16)[:, None],
            torch.stack([torch.linspace(-1, 1, 16), torch.linspace(1, -1, 16) ** 2], dim=1),
            task="regression",
            views=("hidden", "cases", "backprop"),
            backprop_step=1,
            theme=t,
        ),
        lambda t: explore(
            make("three_classes"),
            *clusters([(-1, 1), (1, 1), (0, -1)], [0, 1, 2], noise=0.3, n_per=30),
            views=("backprop", "cases", "hidden"),
            backprop_step=1,
            theme=t,
        ),
    ]
    for name, T in (("dark", DARK_THEME), ("light", THEME)):
        for call in calls:
            s = call(name)
            assert s.fig.get_facecolor()[:3] == pytest.approx(to_rgb(T.paper))
            assert every_text_is_legible(s.fig, T) == [], name
            plt.close("all")


def test_theme_option_switch_and_layout(monkeypatch):
    import IPython.display
    import matplotlib.pyplot as plt
    from helpers import make

    from slopewalk import explore, xor_centres
    from slopewalk.draw import DARK_THEME, THEME

    pytest.importorskip("ipywidgets")
    monkeypatch.delenv("NB_STATIC", raising=False)
    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: None)
    s = explore(make("tanh3"), *xor_centres())
    C = s.controller
    assert C.theme_pick.value == "light" and s.theme.paper == THEME.paper
    png = C.image.value
    C.theme_pick.value = "dark"
    assert s.theme_name == "dark" and s.theme.paper == DARK_THEME.paper and C.image.value != png
    assert s.fig.get_facecolor()[:3] == pytest.approx(to_rgb(DARK_THEME.paper)) and "Colours dark" in C.message.value
    C.layout_pick.value = "standard"  # a new layout keeps the colours
    assert s.theme.paper == DARK_THEME.paper and s.fig.get_facecolor()[:3] == pytest.approx(to_rgb(DARK_THEME.paper))
    assert explore(make("tanh3"), *xor_centres(), theme="dark").controller.theme_pick.value == "dark"
    with pytest.raises(ValueError, match="theme must be 'light' or 'dark'"):
        explore(make("tanh3"), *xor_centres(), theme="purple")
    plt.close("all")


def test_class_colours_differ_from_the_other_meanings():
    """Classes 2 and 3 take hues that no other part of the picture uses: at least 25 degrees from the hidden layer,
    the negative weights, the backward pass and momentum, and from each other class."""
    import colorsys

    from slopewalk.draw import DARK_THEME, THEME

    def hue(c):
        return colorsys.rgb_to_hls(*to_rgb(c))[0] * 360

    def apart(a, b):
        d = abs(hue(a) - hue(b)) % 360
        return min(d, 360 - d)

    for T in (THEME, DARK_THEME):
        assert sorted(T.class_colour) == sorted(T.class_tint) == [0, 1, 2, 3]
        others = [T.hidden, T.negative, T.backprop_colour, T.momentum_colour]
        for k in (2, 3):
            c = T.class_colour[k]
            assert all(apart(c, o) >= 25 for o in others), (k, c)
            assert all(apart(c, d) >= 25 for j, d in T.class_colour.items() if j != k), (k, c)
