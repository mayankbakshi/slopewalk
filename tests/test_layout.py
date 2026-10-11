"""Layout: edge labels do not overlap each other or the boxes, in every example; too many weights: no labels."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn
from helpers import make, sine_data
from render_examples import CASES

from slopewalk import explore, xor_centres
from slopewalk.draw import THEME
from slopewalk.layout import ROW_HEIGHT, TEXT_SCALE, UNIT_SPACING, Layout, overlap_area

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


def boxes_of(fig, ax, label_size=THEME.edge_label_size):
    """Pixel boxes after a draw: (edge labels, the other boxed texts, the glyph rectangles). label_size: the size of
    the edge labels in this figure (the session's theme)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels, others = [], []
    for t in ax.texts:
        if t.get_fontsize() == label_size:
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
    wide = explore(make("wide"), X, y, groups, numbers="all")  # 12 units, 49 weights: the rows grow, the values fit
    counts = [sum(t.get_fontsize() == THEME.edge_label_size for t in ax.texts) for ax in panels(wide.fig)]
    assert counts == [49, 0, 0, 0]  # the weights are the same in every panel: the first panel carries the labels
    few = explore(make("wide"), X, y, groups)  # by default, above 15 weights the lines carry no values
    assert all(not any(t.get_fontsize() == THEME.edge_label_size for t in ax.texts) for ax in panels(few.fig))
    assert not any(t.get_text().startswith("weights as in") for ax in panels(few.fig) for t in ax.texts)
    counts = [sum(t.get_fontsize() == THEME.edge_label_size for t in ax.texts) for ax in panels(values_only.fig)]
    assert counts == [13, 0, 0, 0]  # 13 weights keep their values
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
    assert h3 == pytest.approx(2 * ROW_HEIGHT + 4.2) and h4 > h3 and hsd > h3
    # 4 units: the units stay about 0.9 in apart; 17 weights: the table's two blocks keep 12 pt in a taller first row
    assert h4 == pytest.approx(2 * UNIT_SPACING * 3 + 16 * 12 * 1.45 / 72 / 0.8)
    net = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 1))
    torch.manual_seed(0)
    height = explore(net, X, y, groups).fig.get_size_inches()[1]
    # 12 units: rows four times the usual; 49 weights: the table's two blocks keep 12 pt in a taller first row
    assert height == pytest.approx(2 * 4 * ROW_HEIGHT + 26 * 12 * 1.45 / 72 / 0.8)


def test_every_layer_sits_in_the_middle_of_its_panel(static):
    """The inputs, the units of every layer and the output row are centred on one height, and the panel's range is
    centred on it too, with the bias nodes and the names of the activation curves inside the range."""
    X, y, groups = xor_centres()
    for name in ("tanh3", "relu4", "three_classes", "two_hidden"):
        torch.manual_seed(0)
        net = make(name)
        data = (X, y, groups) if name != "three_classes" else five_centres()
        s = explore(net, *data)
        L = s.layout
        lo, hi = L.ylim
        centre = (lo + hi) / 2
        assert np.mean(L.y_inputs) == pytest.approx(centre)
        for m in list(L.hidden) + [L.outputs]:
            assert np.mean(L.unit_ys(m)) == pytest.approx(centre)
        ax = panels(s.fig)[0]
        s.fig.canvas.draw()
        renderer = s.fig.canvas.get_renderer()
        bias = [t for t in ax.texts if t.get_text() == "1"]
        assert bias and all(lo < t.get_position()[1] < hi for t in bias)
        top = max(t.get_window_extent(renderer).y1 for t in bias)
        names = [t for t in ax.texts if t.get_text().lower() in ("tanh", "relu")]
        bottom = min(t.get_window_extent(renderer).y0 for t in names)
        y0, y1 = ax.get_window_extent(renderer).y0, ax.get_window_extent(renderer).y1
        assert y0 <= bottom and top <= y1  # nothing hangs out of the panel
        plt.close("all")


def five_centres(noise=0.0):
    """The XOR centres and a fifth one of class 2 in the middle."""
    from slopewalk import clusters

    return clusters([(-1, 1), (1, 1), (-1, -1), (1, -1), (0, 0)], [1, 0, 0, 1, 2], noise=noise, n_per=100)


def four_class_centres():
    from slopewalk import clusters

    return clusters([(-1, 1), (1, 1), (-1, -1), (1, -1)], [0, 1, 2, 3])


MORE = {
    "xor_output_only": lambda: (make("tanh3"), *xor_centres(), dict(params="output")),
    "xor_all_views": lambda: (make("tanh3"), *xor_centres(), dict(views=("cases", "slice", "line", "hidden"))),
    "sine_regression": lambda: (
        nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 1)),
        *sine_data(),
        dict(task="regression"),
    ),
    "three_classes": lambda: (make("three_classes"), *five_centres(), {}),
    "three_classes_wide": lambda: (make("three_classes"), *five_centres(), dict(layout="wide")),
    "four_classes": lambda: (make("softmax_only"), *four_class_centres(), {}),
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


def test_wide_layout_sizes(static):
    X, y, groups = xor_centres()
    standard = explore(make("tanh3"), X, y, groups)
    wide = explore(make("tanh3"), X, y, groups, layout="wide")
    assert tuple(standard.fig.get_size_inches()) == pytest.approx((15, 2 * ROW_HEIGHT + 4.2))
    rows = ROW_HEIGHT * TEXT_SCALE["wide"]  # the wide layout has larger text in the network panels, and taller rows
    first = 14 * THEME.table_size * TEXT_SCALE["wide"] * THEME.table_leading / 72 / 0.8  # the 13.2 pt table: 4.65 in
    assert tuple(wide.fig.get_size_inches()) == pytest.approx((27, rows + first))  # four panels in one row
    tops = {round(ax.get_position().y0, 3) for ax in panels(wide.fig)}
    assert len(tops) == 1 and len(panels(wide.fig)) == 4
    views = explore(make("tanh3"), X, y, groups, layout="wide", views=("slice", "line", "momentum"))
    lower = views.fig.axes[4:]
    rows = {ax.get_subplotspec().get_topmost_subplotspec().rowspan.start for ax in lower}
    assert len(lower) == 6 and len(rows) == 1  # one row of six
    assert explore(make("tanh3"), X, y, groups, width=18).fig.get_size_inches()[0] == 18
    deep = explore(make("two_hidden"), X, y, groups, layout="wide")
    assert len({round(ax.get_position().y0, 3) for ax in panels(deep.fig)}) == 2  # deep networks: two per row
    with pytest.raises(ValueError, match="layout must be"):
        explore(make("tanh3"), X, y, groups, layout="tall")
    with pytest.raises(ValueError, match="width is the figure width"):
        explore(make("tanh3"), X, y, groups, width=200)


WIDE = {
    "xor_wide": lambda: (make("tanh3"), *xor_centres()),
    "relu4_wide": lambda: (make("relu4", seed=1), *xor_centres()),
    "clusters_wide": lambda: (make("tanh3"), *xor_centres(noise=0.35)),
}


@pytest.mark.parametrize("name", list(WIDE))
def test_edge_labels_do_not_overlap_in_the_wide_layout(name, static):
    net, X, y, groups = WIDE[name]()
    s = explore(net, X, y, groups, layout="wide", numbers="all")  # values on the lines whatever the count
    for c, ax in enumerate(panels(s.fig)):
        labels, others, glyphs = boxes_of(s.fig, ax, s.theme.edge_label_size)
        assert len(labels) == (len(s.names) if c == 0 else 0)  # the first panel carries every weight's label
        for i, a in enumerate(labels):
            area = (a[2] - a[0]) * (a[3] - a[1])
            for j, b in enumerate(labels):
                if j != i:
                    assert overlap_area(a, b) <= 0.02 * area, (name, ax.get_title(), i, j)
            for b in others + glyphs:
                assert overlap_area(a, b) <= 0.05 * area, (name, ax.get_title(), i)


@pytest.mark.parametrize("step", [2, 3, 5, 6])
@pytest.mark.parametrize("style", ["standard", "wide"])
def test_backprop_labels_do_not_overlap(step, style, static):
    from render_examples import reference_net

    s = explore(reference_net(40), *xor_centres(), views="backprop", backprop_step=step, layout=style, numbers="all")
    expected = {2: [4] * 4, 3: [3, 0, 0, 0], 5: [9] * 4, 6: [13] * 4}[step]  # per-case slopes go in every panel
    assert [len(boxes_of(s.fig, ax, s.theme.edge_label_size)[0]) for ax in panels(s.fig)] == expected
    for ax in panels(s.fig):
        labels, others, glyphs = boxes_of(s.fig, ax, s.theme.edge_label_size)
        for i, a in enumerate(labels):
            area = (a[2] - a[0]) * (a[3] - a[1])
            for j, b in enumerate(labels):
                if j != i:
                    assert overlap_area(a, b) <= 0.02 * area, (step, style, ax.get_title(), i, j)
            for b in others + glyphs:
                assert overlap_area(a, b) <= 0.05 * area, (step, style, ax.get_title(), i)


def test_network_panels_keep_their_size_and_spacing(static):
    """The panels below the networks do not change the network panels: half an inch between them, in every case."""
    X, y, groups = xor_centres()
    for style in ("standard", "wide"):
        gaps, widths = set(), []
        for views in ((), ("backprop",), ("slice", "line"), ("cases", "slice", "line", "hidden", "momentum")):
            fig = explore(make("tanh3"), X, y, groups, layout=style, views=views).fig
            W = fig.get_figwidth()
            boxes = [ax.get_position() for ax in panels(fig)]
            row = [b for b in boxes if abs(b.y0 - boxes[0].y0) < 1e-6]
            gaps |= {round((b.x0 - a.x1) * W, 1) for a, b in zip(row, row[1:])}
            widths.append(boxes[0].width * W)
            plt.close("all")
        assert gaps == {0.5} and max(widths) - min(widths) < 0.15 * max(widths)


def test_lower_rows():
    wide = Layout(2, [3], style="wide")
    (row,) = wide.lower_rows(["plane", "history", "table"])
    widths = dict(row)
    assert list(widths) == ["plane", "history", "table"] and sum(widths.values()) == pytest.approx(27)
    assert widths["plane"] == 4.6 and widths["history"] > widths["table"] > widths["plane"]  # R gets the spare width
    rows = wide.lower_rows(["plane", "history", "table", "backprop", "cases", "slice", "line", "hidden", "momentum"])
    assert [[n for n, _ in r] for r in rows] == [
        ["plane", "history", "table", "backprop"],
        ["cases", "slice", "line", "hidden", "momentum"],
    ]
    assert all(sum(w for _, w in r) == pytest.approx(27) for r in rows)  # every row is filled
    standard = Layout(2, [3])
    assert standard.lower_rows(["plane", "history", "table", "backprop", "cases"]) == [
        [("plane", 5.0), ("history", 5.0), ("table", 5.0)],
        [("backprop", 15.0)],
        [("cases", 5.0), (None, 5.0), (None, 5.0)],
    ]
    # more than 15 weights: the table's two blocks take 7.5 inches of their row in both layouts
    assert standard.lower_rows(["plane", "history", "table", "backprop"], n_weights=27) == [
        [("plane", 3.75), ("history", 3.75), ("table", 7.5)],
        [("backprop", 15.0)],
    ]
    assert standard.table_blocks(27) == (2, pytest.approx((7.5 - 0.8) * 0.97 / 2))
    rows = wide.lower_rows(["plane", "history", "table", "backprop", "slice"], n_weights=27)
    assert [[n for n, _ in r] for r in rows] == [["plane", "history", "table"], ["backprop", "slice"]]
    assert dict(rows[0])["table"] >= 7.5 and all(sum(w for _, w in r) == pytest.approx(27) for r in rows)


def test_table_font_with_two_blocks(static):
    """More than 15 weights: the table keeps a 12 pt font in both layouts (it was 8 pt in a third of the standard
    figure), the first row below the networks grows to fit its rows, and the notes of R over the steps take two
    lines each in the narrower panel beside it."""
    torch.manual_seed(0)
    for style, size in (("standard", 12.0), ("wide", 12.5)):
        s = explore(make("three_classes"), *five_centres(), layout=style)
        s.step(1)
        fig = s.draw()
        table = [ax for ax in fig.axes if ax.tables][0]
        sizes = {round(c.get_text().get_fontsize(), 1) for c in table.tables[0].get_celld().values()}
        assert max(sizes) == pytest.approx(size)
        history = [ax for ax in fig.axes if ax.get_title().startswith("R = ")][0]
        note = [t for t in history.texts if t.get_text().startswith("next step")][0]
        assert ("\npredicted change" in note.get_text()) == (style == "standard")
        plt.close("all")


BOXES = {
    "xor": lambda: (make("tanh3"), *xor_centres(), {}),
    "clusters": lambda: (make("tanh3"), *xor_centres(noise=0.35), {}),
    "relu4": lambda: (make("relu4", seed=1), *xor_centres(), {}),
    "two_hidden": lambda: (make("two_hidden"), *xor_centres(), {}),
    "twelve_units": lambda: (make("wide"), *xor_centres(), {}),
    "regression": lambda: (nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 1)), *sine_data(), {}),
    "backprop_3": lambda: (make("tanh3"), *xor_centres(), {"views": "backprop", "backprop_step": 3}),
    "backprop_clusters_5": lambda: (make("tanh3"), *xor_centres(noise=0.35), {"views": "backprop", "backprop_step": 5}),
    "three_classes": lambda: (make("three_classes"), *five_centres(), {}),
    "three_classes_clusters": lambda: (make("three_classes"), *five_centres(0.35), {}),
    "three_classes_backprop": lambda: (
        make("three_classes"),
        *five_centres(),
        {"views": "backprop", "backprop_step": 3},
    ),
    "three_classes_backprop_clusters": lambda: (
        make("three_classes"),
        *five_centres(0.35),
        {"views": "backprop", "backprop_step": 1},
    ),
    "four_classes": lambda: (make("softmax_only"), *four_class_centres(), {"views": "backprop", "backprop_step": 1}),
    "two_targets": lambda: (
        nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 2)),
        *two_targets(),
        {"task": "regression"},
    ),
    "two_targets_backprop": lambda: (
        nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 2)),
        *two_targets(),
        {"task": "regression", "views": "backprop", "backprop_step": 1},
    ),
    "four_targets_single_points": lambda: (
        nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 4)),
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.5]]),
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, -0.6, 0.7, -0.8], [0.0, 1.0, -1.0, 0.5]]),
        None,
        {"task": "regression"},
    ),
}


def two_targets():
    x = torch.linspace(-2, 2, 16)[:, None]
    return x, torch.stack([torch.sin(1.5 * x[:, 0]), torch.cos(1.5 * x[:, 0])], dim=1), None


@pytest.mark.parametrize("style", ["standard", "wide"])
@pytest.mark.parametrize("name", list(BOXES))
def test_boxes_do_not_overlap(name, style, static):
    """The boxed values of a network panel (inputs, z, h, a, p, the loss box, the backward values) and the
    activation curves keep clear of each other, in both layouts and at the text sizes of each."""
    torch.manual_seed(0)
    net, X, y, groups, options = BOXES[name]()
    options = dict(options)
    task = "regression" if name == "regression" else options.pop("task", None)
    s = explore(net, X, y, groups, layout=style, task=task, **options)
    for ax in panels(s.fig):
        _, boxes, glyphs = boxes_of(s.fig, ax, s.theme.edge_label_size)
        for i, a in enumerate(boxes):
            area = (a[2] - a[0]) * (a[3] - a[1])
            for b in boxes[i + 1 :] + glyphs:
                small = min(area, (b[2] - b[0]) * (b[3] - b[1]))
                assert overlap_area(a, b) <= 0.02 * small, (name, style, ax.get_title(), a, b)


def test_wide_text_is_larger(static):
    """The wide picture is shown smaller, so its network panels use larger text, with room made for it."""
    X, y, groups = xor_centres()
    standard = explore(make("tanh3"), X, y, groups, layout="standard")
    wide = explore(make("tanh3"), X, y, groups, layout="wide")
    f = TEXT_SCALE["wide"]
    assert wide.theme.edge_label_size == pytest.approx(standard.theme.edge_label_size * f)
    assert wide.theme.panel_title_size == standard.theme.panel_title_size  # the panels below keep their sizes
    labels = [t for t in panels(wide.fig)[0].texts if t.get_fontsize() == wide.theme.edge_label_size]
    assert len(labels) == 13  # the values still fit on the lines
    assert wide.layout.x_loss > standard.layout.x_loss


@pytest.mark.parametrize("style", ["wide", "standard"])
def test_one_panel_is_a_zoom_not_a_stretch(style, static):
    """A single network panel (one group, or one group shown with focus) keeps the shape of one of two side by side
    and is 1.5 times larger, text included, in the middle of its row."""
    from slopewalk.layout import ZOOM

    torch.manual_seed(0)
    X = torch.rand(30, 3) * 4 - 2
    net = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
    one = explore(net, X, X[:, 0] * X[:, 1], task="regression", layout=style)
    two = explore(net, X, X[:, 0] * X[:, 1], [0] * 15 + [1] * 15, task="regression", layout=style)
    (box,) = [ax.get_position() for ax in panels(one.fig)]
    pair = [ax.get_position() for ax in panels(two.fig)][0]
    W1, H1 = one.fig.get_size_inches()
    W2, H2 = two.fig.get_size_inches()
    assert box.width * W1 == pytest.approx(pair.width * W2 * ZOOM, rel=0.08)
    assert box.height * H1 == pytest.approx(pair.height * H2 * ZOOM, rel=0.08)
    assert abs((box.x0 + box.x1) / 2 - 0.5) < 0.03
    assert one.theme.edge_label_size == pytest.approx(two.theme.edge_label_size * ZOOM, abs=0.01)
