"""Multiclass: a network with K outputs, the scores of the classes 0 to K-1; R is the cross-entropy."""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import MULTICLASS, case_slope_by_offset, class_points, make

from slopewalk import clusters
from slopewalk.data import prepare
from slopewalk.model import Model


@pytest.mark.parametrize("name", list(MULTICLASS))
def test_risk_is_the_cross_entropy(name):
    net = make(name)
    X, y = class_points(net)
    M = Model(net)
    assert M.task == "multiclass" and M.n_outputs == net[-1].out_features
    expected = F.cross_entropy(net(X), y.long()).item()
    assert M.risk(X, y) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("name", list(MULTICLASS))
def test_slopes_match_autograd_and_finite_differences(name):
    net = make(name)
    X, y = class_points(net)
    M = Model(net)
    g = M.slopes(X, y)
    ref = copy.deepcopy(net)
    F.cross_entropy(ref(X), y.long()).backward()
    auto = []  # in the order of the names
    for m in ref:
        if isinstance(m, nn.Linear):
            for j in range(m.out_features):
                auto += m.weight.grad[j].tolist() + [m.bias.grad[j].item()]
    assert g.tolist() == pytest.approx(auto, abs=1e-6)
    M64 = Model(copy.deepcopy(net).double())
    X64, y64 = X.double(), y.double()
    w = M64.get_weights()
    h = 1e-6
    for k in range(len(w)):
        up, down = list(w), list(w)
        up[k] += h
        down[k] -= h
        fd = (M64.risk(X64, y64, up) - M64.risk(X64, y64, down)) / (2 * h)
        assert g[k] == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("name", list(MULTICLASS))
def test_case_slopes_average_to_the_slopes(name):
    net = make(name)
    X, y = class_points(net)
    M = Model(net)
    cases = M.case_slopes(X, y)
    assert cases.shape == (X.shape[0], M.n_weights)
    assert cases.mean(0) == pytest.approx(M.slopes(X, y), abs=1e-6)
    for k in (0, M.n_weights - 1):
        assert cases[:, k] == pytest.approx(case_slope_by_offset(M, X, y, k), abs=1e-6)


def test_risk_batch_matches_risk():
    net = make("three_classes")
    X, y = class_points(net, n=11)
    M = Model(net)
    w = np.asarray(M.get_weights())
    W = w[None, :] + 0.3 * np.random.default_rng(0).standard_normal((5, len(w)))
    batch = M.risk_batch(X, y, W)
    assert batch == pytest.approx([M.risk(X, y, list(row)) for row in W], abs=1e-6)


def test_forward_values_and_accuracy():
    net = make("three_classes")
    X, y = class_points(net, n=12)
    M = Model(net)
    out = M.forward(X, y)
    a = net(X).detach()
    assert out.a.shape == (12, 3) and out.p.shape == (12, 3)
    assert out.a == pytest.approx(a.numpy(), abs=1e-6)
    assert out.p.sum(1) == pytest.approx(np.ones(12), abs=1e-6)
    assert out.loss == pytest.approx(F.cross_entropy(a, y.long(), reduction="none").numpy(), abs=1e-6)
    assert M.accuracy(X, y) == pytest.approx(float((a.argmax(1) == y.long()).float().mean()))


def test_names_and_titles():
    M = Model(make("three_classes"))
    assert M.n_weights == 9 + 12 and M.names[-1] == "w21"
    assert [M.unit_title(1, j) for j in range(3)] == [f"output unit, class {j}" for j in range(3)]
    assert Model(make("tanh3")).unit_title(1, 0) == "output unit"


def test_task_from_the_outputs():
    net = make("three_classes")
    X, y = class_points(net)
    assert prepare(X, y, n_inputs=2, n_outputs=3).task == "multiclass"
    D = prepare(X, y, n_inputs=2, n_outputs=3)
    assert D.n_classes == 3
    two = make("two_outputs")
    assert prepare(X, (y > 0).float(), n_inputs=2, n_outputs=2).task == "multiclass"
    assert Model(two).task == "multiclass"
    assert prepare(X, (y > 0).float(), n_inputs=2).n_classes == 2


def test_default_groups_are_the_classes():
    X, y, _ = clusters([(-1, 1), (1, 1), (0, -1)], [0, 1, 2], noise=0.3, n_per=10)
    D = prepare(X, y, n_inputs=2, n_outputs=3)
    assert D.G == 3 and [int(D.y[sel].unique().item()) for sel in D.gsel] == [0, 1, 2]


@pytest.mark.parametrize(
    "y, n_outputs, task, fragment",
    [
        (
            [0, 1, 3],
            3,
            None,
            "label 3, but the network has 3 outputs, one per class: the labels must be whole numbers from 0 to 2",
        ),
        ([0, 1, 1.5], 3, None, "label 1.5"),
        ([0, -1, 1], 3, None, "label -1"),
        ([0, 1, 2], 3, "binary", "has 3 outputs: one score per class is task='multiclass'"),
        ([0, 1, 2], 3, "regression", "the network has 3 outputs, so y needs 3 targets per point"),
        ([0, 1, 1], 1, "multiclass", "needs one output per class"),
        ([0, 1, 2], 1, None, "For 3 classes, end the network with nn.Linear(..., 3) and use the labels 0 to 2"),
    ],
)
def test_labels_and_task_are_checked(y, n_outputs, task, fragment):
    X = [[0, 0], [1, 1], [2, 2]]
    with pytest.raises(ValueError, match="slopewalk: ") as e:
        prepare(X, y, n_inputs=2, task=task, n_outputs=n_outputs)
    assert fragment.lower() in str(e.value).lower()


def test_more_than_four_classes_are_refused():
    with pytest.raises(ValueError, match=r"nn.Linear\(..., 5\), 5 outputs; slopewalk draws up to 4"):
        Model(nn.Sequential(nn.Linear(2, 5)))


def test_float64_network():
    net = make("three_classes").double()
    X, y = class_points(net)
    M = Model(net)
    assert M.risk(X.double(), y.double()) == pytest.approx(F.cross_entropy(net(X.double()), y.long()).item(), abs=1e-12)


def test_one_hot_targets_are_not_needed():
    """The labels stay a float tensor in the session, like binary labels; the loss converts them to class indices."""
    net = make("three_classes")
    X, y = class_points(net)
    D = prepare(X, y, n_inputs=2, n_outputs=3)
    assert D.y.dtype == torch.float32 and set(D.y.tolist()) <= {0.0, 1.0, 2.0}


# ---------------------------------------------------------------- the picture


@pytest.fixture
def static(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def five_centres(noise=0.0):
    """The owner's example: the XOR centres and a fifth centre of class 2 in the middle."""
    return clusters([(-1, 1), (1, 1), (-1, -1), (1, -1), (0, 0)], [1, 0, 0, 1, 2], noise=noise, n_per=100)


def network_panels(fig):
    return [ax for ax in fig.axes if ax.get_title()[:1].isupper() and ": " in ax.get_title()]


def test_network_panels(static):
    from slopewalk import explore
    from slopewalk.draw import THEME

    s = explore(make("three_classes"), *five_centres())
    P = network_panels(s.fig)
    assert len(P) == 5 and P[4].get_title().startswith("E: (0.0, 0.0), class 2:")
    assert P[4].title.get_color() == THEME.class_colour[2]
    texts = {t.get_text().split("\n")[0]: t for t in P[0].texts}
    assert {"a0", "a1", "a2", "p0", "p1", "p2", "softmax"} <= set(texts)
    edges = {k: texts[f"p{k}"].get_bbox_patch().get_edgecolor() for k in range(3)}
    from matplotlib.colors import to_rgba

    assert edges[1] == to_rgba(THEME.class_colour[1])  # panel A is class 1: its p1 carries the class colours
    assert edges[0] == edges[2] == to_rgba(THEME.value_box["ec"])
    assert any(t.get_text().startswith("y = 1\nloss ") for t in P[0].texts)
    assert all(len([ln for ln in ax.lines if ln.get_zorder() == 1]) == 21 for ax in P)


def test_plane_shows_the_class_regions(static):
    from slopewalk import explore

    torch.manual_seed(0)
    s = explore(make("three_classes"), *five_centres())
    plane = [ax for ax in s.fig.axes if ax.get_title() == "the most likely class and the boundaries"][0]
    assert len(plane.images) == 1
    from slopewalk.draw import net_on, plane_grid

    winners = set(net_on(s, plane_grid(s)[2], s.weights()).argmax(1).tolist())
    expected = [f"class {k}" if k in winners else f"class {k}: no region" for k in range(3)]
    assert [t.get_text() for t in plane.get_legend().get_texts()] == expected and len(winners) < 3
    lines_before = len(plane.collections)
    s.step(20, eta=1.0)
    plane = [ax for ax in s.draw().axes if ax.get_title() == "the most likely class and the boundaries"][0]
    assert len(plane.collections) > lines_before  # the starting boundaries, dashed


def test_a_class_gets_its_region_once_learned(static):
    """After 40 steps no point is most likely class 2, and the legend says so; after 300 steps of a network with four
    hidden units, the middle of the plane is class 2."""
    from slopewalk import explore

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 3))
    s = explore(net, *five_centres(0.35))
    legend = lambda fig: [  # noqa: E731
        t.get_text()
        for ax in fig.axes
        if ax.get_legend()
        for t in ax.get_legend().get_texts()
        if "class" in t.get_text()
    ]
    s.step(40, eta=0.5)
    assert "class 2: no region" in legend(s.draw())
    s.step(260, eta=0.5)
    assert legend(s.draw()) == ["class 0", "class 1", "class 2"]


def test_class_shading():
    """The colour of the most likely class, stronger the surer: five bands from paper to plane_alpha of the colour."""
    from slopewalk.draw import THEME, class_shading

    A = np.array([[[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0], [1.0, 1.0, 1.0]]])
    image = class_shading(THEME, A)
    paper = np.asarray(to_rgb(THEME.paper))

    def mix(k, band):
        share = THEME.plane_alpha * band
        return paper * (1 - share) + np.asarray(to_rgb(THEME.class_colour[k])) * share

    for k in range(3):
        assert image[0, k] == pytest.approx(mix(k, 0.9))  # sure: the strongest band
    assert image[0, 3] == pytest.approx(mix(0, 0.1))  # all equally likely: the faintest band


def to_rgb(c):
    from matplotlib.colors import to_rgb as rgb

    return rgb(c)


def test_one_input(static):
    from slopewalk import explore

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 3))
    X, y, g = clusters([(-2,), (0,), (2,)], [0, 1, 2], noise=0.3, n_per=10)
    s = explore(net, X, y, g, views="hidden")
    ax = [a for a in s.fig.axes if a.get_title().startswith("p of each class against x1")][0]
    assert [ln.get_label() for ln in ax.lines[:3]] == ["p0", "p1", "p2"]
    assert any(a.get_title().startswith("the cases at their hidden values") for a in s.fig.axes)


@pytest.mark.parametrize("name", list(MULTICLASS))
def test_every_view_draws(name, static):
    from slopewalk import explore

    net = make(name)
    X, y = class_points(net, n=12)
    views = ("cases", "slice", "line", "hidden", "momentum", "backprop")
    s = explore(net, X, y, views=views, backprop_step=1)
    s.step(2)
    fig = s.draw()
    titles = [ax.get_title() for ax in fig.axes]
    assert any(t.startswith("each case's slope along") and "Σk (pk - yk) dak/dw" in t for t in titles)
    assert any(t.startswith("the cases at their hidden values") for t in titles) or "no hidden layer" in str(
        [t.get_text() for ax in fig.axes for t in ax.texts]
    )


# ---------------------------------------------------------------- backpropagation


@pytest.mark.parametrize("name", list(MULTICLASS))
def test_backward_pass_gives_each_case_slope(name):
    from slopewalk.backprop import backward_pass

    net = make(name)
    X, y = class_points(net)
    M = Model(net)
    bp = backward_pass(M, X, y)
    P = torch.softmax(net(X), dim=1).detach().numpy()
    onehot = np.eye(M.n_outputs)[y.long().numpy()]
    assert bp.delta[-1] == pytest.approx(P - onehot, abs=1e-6)
    assert bp.grads == pytest.approx(M.case_slopes(X, y), abs=1e-6)


def test_stages_and_worked_example(static):
    from slopewalk import explore
    from slopewalk.backprop import derivation_lines

    s = explore(make("three_classes"), *five_centres(), views="backprop", backprop_step=1)
    assert s.backprop_stages[1].text.startswith("at the output: δak = ∂ℓ/∂ak = pk - yk")
    i = 0  # case A, class 1
    lines = [t for _, t in derivation_lines(s, s.picture.frame if hasattr(s, "picture") else make_frame(s), i)]
    a = s.model.forward(s.data.X, s.data.y).a[i]
    p = np.exp(a - a.max()) / np.exp(a - a.max()).sum()
    assert lines[0] == f"forward: a = ({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}), y = 1"
    assert lines[1].startswith("p = softmax(a) = (")
    assert lines[2:5] == [
        f"δa0 = p0 - 0 = {p[0]:.3f} - 0 = {p[0]:+.3f}",
        f"δa1 = p1 - 1 = {p[1]:.3f} - 1 = {p[1] - 1:+.3f}",
        f"δa2 = p2 - 0 = {p[2]:.3f} - 0 = {p[2]:+.3f}",
    ]
    assert any(t.startswith("w10: δa0 × h1 = ") for t in lines)
    panel = network_panels(s.fig)[0]
    deltas = [t.get_text() for t in panel.texts if t.get_text().startswith("δa")]
    assert [d.split("\n")[0] for d in deltas] == ["δa0", "δa1", "δa2"]


def make_frame(s):
    from slopewalk.picture import make_frame as frame

    return frame(s)


# ---------------------------------------------------------------- the controls


def test_controls(monkeypatch):
    ipywidgets = pytest.importorskip("ipywidgets")
    import IPython.display

    from slopewalk import explore

    monkeypatch.delenv("NB_STATIC", raising=False)
    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: None)
    s = explore(make("three_classes"), *five_centres(), views="backprop")
    C = s.controller
    assert "accuracy" in C.readout.value
    heads = [w.children[0].value for w in C.panel.children if isinstance(w, ipywidgets.VBox)]
    assert heads[-3:] == [f"<b>output unit, class {k}</b>: weights for h1, h2, h3, bias" for k in range(3)]
    C.bp_next.click()
    assert "Under each score ak, the purple box is δak = pk - yk" in C.message.value
    C.buttons["gradient step"].click()
    assert len(s.history) == 2 and s.history[1] < s.history[0]


@pytest.mark.parametrize("style", ["wide", "standard"])
def test_worked_example_lines_fit_their_column(style, static):
    """Sums over three or more next units go two terms to a line, so that every line of the Worked example stays in
    its column (the owner's wide call with the slice beside it)."""
    from slopewalk import explore

    s = explore(make("three_classes"), *five_centres(), layout=style, views=("backprop", "slice"), backprop_step=3)
    fig = s.fig
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = [a for a in fig.axes if any(t.get_text() == "Worked example" for t in a.texts)][0]
    panel = ax.get_window_extent(renderer)
    texts = [t for t in ax.texts if t.get_text() != "Worked example" and not t.get_text().startswith(":")]
    assert any(t.get_text().startswith("      + ") for t in texts)  # a continued sum
    for t in texts:
        box = t.get_window_extent(renderer)
        first = box.x0 < panel.x0 + 0.5 * panel.width - 1
        assert box.x1 <= panel.x0 + (0.5 if first else 1.0) * panel.width + 1, t.get_text()
