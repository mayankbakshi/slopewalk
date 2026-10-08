"""Layout: edge labels do not overlap each other or the boxes, in every example; too many weights: no labels."""

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from helpers import make
from render_examples import CASES

from slopewalk import explore, xor_centres
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
    none = explore(make("wide"), X, y, groups)
    assert all(not any(t.get_fontsize() == THEME.edge_label_size for t in ax.texts) for ax in panels(none.fig))
    table = [tbl for ax in none.fig.axes for tbl in ax.tables]
    assert sum(max(k[0] for k in t.get_celld()) for t in table) == 49  # every weight's value is in the table


def test_rows_grow_with_units_and_groups(static):
    X, y, groups = xor_centres()
    h3 = explore(make("tanh3"), X, y, groups).fig.get_size_inches()[1]
    h4 = explore(make("relu4"), X, y, groups).fig.get_size_inches()[1]
    hsd = explore(make("tanh3"), *xor_centres(noise=0.3)).fig.get_size_inches()[1]
    assert h3 == pytest.approx(2 * 2.9 + 4.2) and h4 > h3 and hsd > h3
    net = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 1))
    torch.manual_seed(0)
    assert explore(net, X, y, groups).fig.get_size_inches()[1] <= 2 * 2 * 2.9 + 4.2  # capped at twice the row height
