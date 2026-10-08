"""Layout: edge labels do not overlap each other or the boxes, in every example; too many weights: no labels."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn
from helpers import make
from render_examples import CASES

from slopewalk import explore, sine_points, xor_centres
from slopewalk.draw import THEME
from slopewalk.layout import overlap_area

EXTRA = {
    "three_inputs": lambda: (make("three_inputs"), [(0, 0, 1), (1, 1, 1), (1, 0, 0)], [0, 1, 1], None),
    "wide_49_weights": lambda: (make("wide"), *xor_centres()),
    "relu4_clusters": lambda: (make("relu4"), *xor_centres(noise=0.35)),
}
ALL = {name: build for name, (build, _) in CASES.items()} | EXTRA


@pytest.fixture
def static(monkeypatch):
    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def panels(fig):
    return [ax for ax in fig.axes if ax.get_title()[:1].isupper() and ": " in ax.get_title()]


def boxes_of(fig, ax):
    """Pixel boxes after a draw: (edge labels, the other boxed texts, the glyph rectangles)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels, others = [], []
    for t in ax.texts:
        if t.get_fontsize() == THEME.edge_label_size:
            labels.append(t.get_window_extent(renderer).extents)
        elif t.get_bbox_patch() is not None:
            others.append(t.get_bbox_patch().get_window_extent(renderer).extents)
    glyphs = [p.get_window_extent(renderer).extents for p in ax.patches if abs(p.get_width() - 0.62) < 1e-9]
    return labels, others, glyphs


@pytest.mark.parametrize("name", list(ALL))
def test_edge_labels_do_not_overlap(name, static):
    net, X, y, groups = ALL[name]()
    s = explore(net, X, y, groups)
    for ax in panels(s.fig):
        labels, others, glyphs = boxes_of(s.fig, ax)
        assert len(labels) in (0, len(s.names))
        for i, a in enumerate(labels):
            area = (a[2] - a[0]) * (a[3] - a[1])
            for j, b in enumerate(labels):
                if j != i:
                    assert overlap_area(a, b) <= 0.02 * area, (name, ax.get_title(), i, j)
            for b in others + glyphs:
                assert overlap_area(a, b) <= 0.05 * area, (name, ax.get_title(), i)


def test_label_modes(static):
    X, y, groups = xor_centres()
    with_names = explore(make("logistic"), X, y, groups)
    labels = [t.get_text() for t in panels(with_names.fig)[0].texts if t.get_fontsize() == THEME.edge_label_size]
    assert len(labels) == 3 and all(label.startswith("w") and " = " in label for label in labels)
    values_only = explore(make("tanh3"), X, y, groups)
    labels = [t.get_text() for t in panels(values_only.fig)[0].texts if t.get_fontsize() == THEME.edge_label_size]
    assert len(labels) == 13 and not any(label.startswith("w") for label in labels)
    wide = explore(make("wide"), X, y, groups)  # 12 units, 49 weights: the rows grow, the values still fit
    assert all(sum(t.get_fontsize() == THEME.edge_label_size for t in ax.texts) == 49 for ax in panels(wide.fig))
    torch.manual_seed(0)
    crowded = nn.Sequential(nn.Linear(2, 20), nn.Tanh(), nn.Linear(20, 1))  # 81 weights: no room for labels
    none = explore(crowded, X, y, groups)
    assert all(not any(t.get_fontsize() == THEME.edge_label_size for t in ax.texts) for ax in panels(none.fig))
    table = [tbl for ax in none.fig.axes for tbl in ax.tables]
    assert sum(max(k[0] for k in t.get_celld()) for t in table) == 81  # every weight's value is in the table


def test_rows_grow_with_units_and_groups(static):
    X, y, groups = xor_centres()
    h3 = explore(make("tanh3"), X, y, groups).fig.get_size_inches()[1]
    h4 = explore(make("relu4"), X, y, groups).fig.get_size_inches()[1]
    hsd = explore(make("tanh3"), *xor_centres(noise=0.3)).fig.get_size_inches()[1]
    assert h3 == pytest.approx(2 * 2.9 + 4.2) and h4 > h3 and hsd > h3
    net = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 1))
    torch.manual_seed(0)
    height = explore(net, X, y, groups).fig.get_size_inches()[1]
    assert height == pytest.approx(2 * 4 * 2.9 + 4.2)  # 12 units: rows four times the usual height


MORE = {
    "xor_output_only": lambda: (make("tanh3"), *xor_centres(), dict(params="output")),
    "xor_all_views": lambda: (make("tanh3"), *xor_centres(), dict(views=("cases", "slice", "line", "hidden"))),
    "sine_regression": lambda: (
        nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 1)),
        *sine_points(),
        dict(task="regression"),
    ),
}


@pytest.mark.parametrize("name", list(MORE))
def test_edge_labels_do_not_overlap_with_options(name, static):
    torch.manual_seed(0)
    net, X, y, groups, kw = MORE[name]()
    s = explore(net, X, y, groups, **kw)
    for ax in panels(s.fig):
        labels, others, glyphs = boxes_of(s.fig, ax)
        assert len(labels) in (0, len(s.names))
        for i, a in enumerate(labels):
            area = (a[2] - a[0]) * (a[3] - a[1])
            for j, b in enumerate(labels):
                if j != i:
                    assert overlap_area(a, b) <= 0.02 * area, (name, ax.get_title(), i, j)
            for b in others + glyphs:
                assert overlap_area(a, b) <= 0.05 * area, (name, ax.get_title(), i)


def test_vectorised_placement_equals_the_original():
    from helpers import place_labels_reference

    from slopewalk.layout import place_labels

    rng = np.random.default_rng(0)
    for _ in range(30):
        m = int(rng.integers(1, 12))
        edges = [(tuple(rng.uniform(0, 4, 2)), tuple(rng.uniform(0, 4, 2))) for _ in range(m)]
        sizes = [tuple(rng.uniform(0.2, 0.6, 1)) + tuple(rng.uniform(0.08, 0.2, 1)) for _ in range(m)]
        x0 = rng.uniform(0, 4, (8, 2))
        obstacles = [(a, b, a + rng.uniform(0.2, 0.8), b + rng.uniform(0.2, 0.8)) for a, b in x0]
        defaults = list(rng.uniform(0.2, 0.8, m))
        new, _, _, _ = place_labels(edges, sizes, obstacles, defaults, pad=0.02)
        old = place_labels_reference(edges, sizes, obstacles, defaults, pad=0.02)
        assert np.allclose(new, old, atol=1e-12)
