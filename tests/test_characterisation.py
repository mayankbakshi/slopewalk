"""Characterisation tests: pin the numbers and the drawn text of the original one-file gradviz.py, before any change.

Two kinds of pins. (1) Numbers computed directly with torch (the oracle): R = 1.003 and slopes (0.616, 0.116, 0.116)
for the two-point start, and R = 0.568 on the four XOR centres at step 40 of the reference run (the 2-3-1 tanh network
from seed 0, 40 full-batch steps with eta 0.5 on the XOR data). (2) The text that explore draws in still-picture mode
(NB_STATIC=1): the title of the history panel, the slope table, the edge labels and the per-case boxes.
"""

import inspect
import re
import sys

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from render_examples import CASES, CENTERS, LABELS, reference_data, reference_net, two_point

from slopewalk import clusters, explore
from slopewalk.model import parse_layers

NAMES13 = [f"w{k}" for k in range(1, 14)]


# ---------------------------------------------------------------- helpers


def flat(net):
    """Weights in the order of their names: layer by layer, unit by unit, inputs in order, then the bias."""
    out = []
    for m in net:
        if isinstance(m, nn.Linear):
            for j in range(m.out_features):
                out += m.weight[j].detach().tolist() + [m.bias[j].item()]
    return out


def oracle(net, X, y):
    """R, its gradient in the order of the names, and a, p, loss per point, all straight from torch."""
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    net.zero_grad()
    R = F.binary_cross_entropy_with_logits(net(X).reshape(-1), y)
    R.backward()
    g = []
    for m in net:
        if isinstance(m, nn.Linear):
            for j in range(m.out_features):
                g += m.weight.grad[j].tolist() + [m.bias.grad[j].item()]
    net.zero_grad()
    with torch.no_grad():
        a = net(X).reshape(-1)
        p = torch.sigmoid(a)
        loss = F.binary_cross_entropy_with_logits(a, y, reduction="none")
    return R.item(), g, a.tolist(), p.tolist(), loss.tolist()


def plus_point_one(net, X, y):
    """The change of R when each weight in turn grows by 0.1 (the first column of the slope table)."""
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    R0 = F.binary_cross_entropy_with_logits(net(X).reshape(-1), y).item()
    out = []
    with torch.no_grad():
        for m in net:
            if isinstance(m, nn.Linear):
                for j in range(m.out_features):
                    for t, i in [(m.weight, (j, i)) for i in range(m.in_features)] + [(m.bias, (j,))]:
                        t[i] += 0.1
                        out.append(F.binary_cross_entropy_with_logits(net(X).reshape(-1), y).item() - R0)
                        t[i] -= 0.1
    return out


def texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts] + [t.get_text() for t in fig.texts]


def edge_labels(fig):
    """The texts on the weight lines: the only texts drawn at 8.5 pt."""
    return [t.get_text() for ax in fig.axes for t in ax.texts if t.get_fontsize() == 8.5]


def history_title(fig):
    titles = [ax.get_title() for ax in fig.axes if ax.get_title().startswith("R = ")]
    assert len(titles) == 1, titles
    return titles[0]


def slope_table(fig):
    """{name: (change of R for +0.1, slope)} read from the cells of the table panel, in row order."""
    rows = {}
    for ax in fig.axes:
        for tbl in ax.tables:
            cells = tbl.get_celld()
            for r in range(1, max(key[0] for key in cells) + 1):
                name, value, delta, slope = (cells[r, c].get_text().get_text() for c in range(4))
                assert re.fullmatch(r"w\d+", name) and re.fullmatch(r"[+-]\d+\.\d{2}", value)
                assert re.fullmatch(r"[+-]\d+\.\d{4}", delta) and re.fullmatch(r"[+-]\d+\.\d{3}", slope)
                rows[name] = (float(delta), float(slope))
    return rows


def panel_titles(fig):
    return [ax.get_title() for ax in fig.axes if re.match(r"^[A-Z]: ", ax.get_title())]


@pytest.fixture
def static(monkeypatch):
    """Runs explore in still-picture mode and returns the figure."""
    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    def run(net, X, y, groups=None, **kw):
        plt.close("all")
        explore(net, X, y, groups, **kw)
        return plt.gcf()

    yield run
    plt.close("all")


# ---------------------------------------------------------------- the reference numbers


def test_two_point_start_numbers():
    net, X, y, _ = two_point()
    R, g, a, p, loss = oracle(net, X, y)
    assert R == pytest.approx(1.003, abs=5e-4)
    assert R == pytest.approx(1.003204, abs=1e-4)
    assert g == pytest.approx([0.616, 0.116, 0.116], abs=5e-4)
    assert g == pytest.approx([0.615529, 0.115529, 0.115529], abs=1e-4)
    assert a == pytest.approx([0.0, 1.0], abs=1e-6)
    assert p == pytest.approx([0.5, 0.731059], abs=1e-4)


def test_reference_run_step40_weights_and_risk():
    net = reference_net(40)
    # the 13 weights of step 40, w1 to w13; they depend on torch's RNG (seed 0) and on the data generator
    expected = [
        0.212207,
        0.513153,
        0.077719,
        -0.920734,
        -0.901621,
        1.050028,
        -0.344259,
        0.077888,
        0.002624,
        0.485262,
        0.972129,
        -0.194198,
        -0.373887,
    ]
    assert flat(net) == pytest.approx(expected, abs=1e-4)
    Xd, yd = reference_data()
    assert oracle(net, Xd, yd)[0] == pytest.approx(0.591216, abs=1e-4)  # R on the 400 points
    Xc, yc, gc = clusters(CENTERS, LABELS)
    R, g, a, p, loss = oracle(net, Xc, yc)
    assert R == pytest.approx(0.568, abs=5e-4)  # R on the four centres (the reference value)
    assert R == pytest.approx(0.567978, abs=1e-4)
    assert a == pytest.approx([0.4908, -0.6310, 0.2644, 0.3493], abs=5e-4)
    assert p == pytest.approx([0.6203, 0.3473, 0.5657, 0.5864], abs=5e-4)
    assert loss == pytest.approx([0.4775, 0.4266, 0.8341, 0.5337], abs=5e-4)


def test_random_start_seed0():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    expected = [
        -0.005294,
        0.379323,
        -0.014010,
        -0.581981,
        -0.520387,
        0.560658,
        -0.272345,
        0.189616,
        -0.062752,
        0.152774,
        -0.174483,
        -0.113487,
        -0.551571,
    ]
    assert flat(net) == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------- clusters


def test_clusters_reproduces_the_reference_data():
    X, y, groups = clusters([(-1, -1), (1, 1), (-1, 1), (1, -1)], [0, 0, 1, 1], noise=0.35, n_per=100, seed=0)
    Xd, yd = reference_data()
    assert torch.equal(X, Xd) and torch.equal(y, yd)
    assert groups.tolist() == [k for k in range(4) for _ in range(100)]


def test_clusters_without_noise_returns_the_centres():
    X, y, groups = clusters(CENTERS, LABELS)
    assert X.dtype == torch.float32 and y.dtype == torch.float32 and groups.dtype == torch.int64
    assert X.tolist() == [[-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
    assert y.tolist() == [1.0, 0.0, 0.0, 1.0] and groups.tolist() == [0, 1, 2, 3]


def test_clusters_is_reproducible_and_seed_dependent():
    a = clusters(CENTERS, LABELS, noise=0.3, n_per=7, seed=3)
    b = clusters(CENTERS, LABELS, noise=0.3, n_per=7, seed=3)
    c = clusters(CENTERS, LABELS, noise=0.3, n_per=7, seed=4)
    assert torch.equal(a[0], b[0]) and not torch.equal(a[0], c[0])
    assert a[0].shape == (28, 2) and a[1].shape == (28,) and a[2].tolist() == [k for k in range(4) for _ in range(7)]


# ---------------------------------------------------------------- model parsing


@pytest.mark.parametrize("name", list(CASES))
def test_layers_accepts_the_example_networks(name):
    net = CASES[name][0]()[0]
    layers = parse_layers(net)
    assert [L.linear for L in layers] == [m for m in net if isinstance(m, nn.Linear)]
    assert layers[-1].act is None and layers[-1].out_features == 1


def test_layers_skips_identity_and_reads_the_activation():
    net = nn.Sequential(nn.Linear(2, 2), nn.Identity(), nn.ReLU(), nn.Linear(2, 1))
    layers = parse_layers(net)
    assert len(layers) == 2 and layers[0].act[0] == "ReLU" and layers[0].module is net[2]


@pytest.mark.parametrize(
    "net, fragment",
    [
        (nn.Sequential(nn.Linear(2, 1), nn.Sigmoid()), "end with nn.Linear(..., 1)"),
        (nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2)), "end with nn.Linear(..., 1)"),
        (nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.1), nn.Linear(3, 1)), "unsupported layer"),
        (nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Tanh(), nn.Linear(3, 1)), "unsupported layer"),
        (nn.Sequential(nn.Tanh(), nn.Linear(2, 1)), "unsupported layer"),
    ],
)
def test_layers_rejects_unsupported_networks(net, fragment):
    with pytest.raises(ValueError, match=re.escape(fragment)):
        parse_layers(net)


def test_public_signatures_unchanged():
    assert str(inspect.signature(explore)) == "(net, X, y, groups=None, eta=1.0, title=None)"
    assert str(inspect.signature(clusters)) == "(centers, labels, noise=0.0, n_per=50, seed=0)"


# ---------------------------------------------------------------- what the still picture shows


def test_two_point_picture(static, capsys):
    net, X, y, _ = two_point()
    fig = static(net, X, y)
    assert "Still picture of the start" in capsys.readouterr().out
    assert history_title(fig) == "R = 1.003"
    table = slope_table(fig)
    assert list(table) == ["w1", "w2", "w3"]
    assert [table[k][1] for k in table] == [0.616, 0.116, 0.116]
    assert [table[k][0] for k in table] == pytest.approx(plus_point_one(net, X, y), abs=1.5e-4)
    # names and values on the lines (at most 9 weights), once per panel
    assert sorted(edge_labels(fig)) == sorted(["w1 = 0.50", "w2 = 0.50", "w3 = 0.00"] * 2)
    assert panel_titles(fig) == ["A: (-1.0, 1.0), class 1:  loss 0.693", "B: (1.0, 1.0), class 0:  loss 1.313"]
    assert "y = 1\nloss 0.693\nchange\nfrom start\n+0.000" in texts(fig)
    assert "y = 0\nloss 1.313\nchange\nfrom start\n+0.000" in texts(fig)
    assert "a\n+0.00" in texts(fig) and "a\n+1.00" in texts(fig)
    assert "p\n0.50" in texts(fig) and "p\n0.73" in texts(fig)
    assert len(fig.axes) == 2 + 3


def test_xor_step40_picture(static):
    net = reference_net(40)
    Xc, yc, gc = clusters(CENTERS, LABELS)
    R, g, a, p, loss = oracle(net, Xc, yc)
    delta = plus_point_one(net, Xc, yc)
    fig = static(net, Xc, yc, gc)
    assert history_title(fig) == "R = 0.568"
    table = slope_table(fig)
    assert list(table) == NAMES13  # the slope table follows the naming order
    assert [table[k][1] for k in NAMES13] == pytest.approx(g, abs=1.5e-3)
    assert [table[k][0] for k in NAMES13] == pytest.approx(delta, abs=1.5e-4)
    # more than 9 weights: values only on the lines, each weight once per panel
    w = flat(net)
    assert sorted(edge_labels(fig)) == sorted([f"{v:.2f}" for v in w] * 4)
    # panels in the order of the centres: A top left, B top right, C bottom left, D bottom right
    assert panel_titles(fig) == [
        f"A: (-1.0, 1.0), class 1:  loss {loss[0]:.3f}",
        f"B: (1.0, 1.0), class 0:  loss {loss[1]:.3f}",
        f"C: (-1.0, -1.0), class 0:  loss {loss[2]:.3f}",
        f"D: (1.0, -1.0), class 1:  loss {loss[3]:.3f}",
    ]
    for i, yi in enumerate(LABELS):
        assert f"y = {yi}\nloss {loss[i]:.3f}\nchange\nfrom start\n+0.000" in texts(fig)
        assert f"a\n{a[i]:+.2f}" in texts(fig) and f"p\n{p[i]:.2f}" in texts(fig)
    assert len(fig.axes) == 4 + 3


def test_group_values_are_mean_and_sd(static):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    X, y, groups = clusters(CENTERS, LABELS, noise=0.35)
    fig = static(net, X, y, groups)
    titles = panel_titles(fig)
    assert len(titles) == 4 and all("mean of 50 points" in t for t in titles)
    xm = X[groups == 0].mean(0)
    assert titles[0].startswith(f"A: mean of 50 points ({xm[0]:.1f}, {xm[1]:.1f}), class 1:")
    with torch.no_grad():
        p = torch.sigmoid(net(X).reshape(-1))
    p0 = p[groups == 0]
    assert f"p\n{p0.mean():.2f}\n±{p0.numpy().std():.2f}" in texts(fig)  # mean ± standard deviation over the group


@pytest.mark.parametrize("name", list(CASES))
def test_still_picture_smoke(static, name):
    build, panels = CASES[name]
    net, X, y, groups = build()
    fig = static(net, X, y, groups)
    assert len(fig.axes) == panels + 3
    assert len(panel_titles(fig)) == panels
    assert history_title(fig).startswith("R = ")
    assert len(slope_table(fig)) == len(flat(net))


def test_title_and_three_inputs(static):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 1))
    fig = static(net, [(0, 0, 1), (1, 1, 1), (1, 0, 0)], [0, 1, 1], title="three inputs")
    assert fig._suptitle.get_text() == "three inputs"
    assert "3 inputs: no picture of the input plane" in texts(fig)
    assert len(slope_table(fig)) == 11


def test_without_ipywidgets_draws_and_explains(monkeypatch, capsys):
    monkeypatch.delenv("NB_STATIC", raising=False)
    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    net, X, y, _ = two_point()
    plt.close("all")
    explore(net, X, y)
    out = capsys.readouterr().out
    assert "No sliders: the package ipywidgets is missing" in out
    assert history_title(plt.gcf()) == "R = 1.003"
    plt.close("all")


# ---------------------------------------------------------------- the controller (ipywidgets without a front end)


def widget_tree(roots):
    seen = []
    stack = list(roots)
    while stack:
        w = stack.pop(0)
        seen.append(w)
        stack += list(getattr(w, "children", ()))
    return seen


@pytest.fixture
def controller(monkeypatch):
    """Runs explore with ipywidgets and no front end; returns (sliders by name, buttons by label, eta slider)."""
    import IPython.display
    import ipywidgets as widgets

    monkeypatch.delenv("NB_STATIC", raising=False)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    shown = []
    monkeypatch.setattr(IPython.display, "display", lambda *objs, **k: shown.extend(objs))

    def run(net, X, y, groups=None, **kw):
        plt.close("all")
        explore(net, X, y, groups, **kw)
        tree = widget_tree(shown)
        sliders = {w.description: w for w in tree if isinstance(w, widgets.FloatSlider)}
        buttons = {w.description: w for w in tree if isinstance(w, widgets.Button)}
        return sliders, buttons

    yield run
    plt.close("all")


def test_controller_sliders_buttons_and_steps(controller):
    net, X, y, _ = two_point()
    sliders, buttons = controller(net, X, y)
    assert set(sliders) == {"w1", "w2", "w3", "eta"} and set(buttons) == {"gradient step", "10 steps", "reset"}
    assert [sliders[k].value for k in ["w1", "w2", "w3"]] == [0.5, 0.5, 0.0]
    assert (sliders["w1"].min, sliders["w1"].max, sliders["w1"].step) == (-4, 4, 0.001)
    assert sliders["eta"].value == 1.0
    # a slider changes the network itself
    sliders["w1"].value = 0.6
    assert net[0].weight[0, 0].item() == pytest.approx(0.6)
    assert history_title(plt.gcf()).startswith("R = ")
    sliders["w1"].value = 0.5
    # one gradient step: w <- w - eta * dR/dw
    _, g, *_ = oracle(net, X, y)
    buttons["gradient step"].click()
    assert flat(net) == pytest.approx([0.5 - g[0], 0.5 - g[1], 0.0 - g[2]], abs=1e-6)
    assert [sliders[k].value for k in ["w1", "w2", "w3"]] == pytest.approx(flat(net), abs=1e-6)
    # reset returns to the weights at call time
    buttons["reset"].click()
    assert flat(net) == pytest.approx([0.5, 0.5, 0.0], abs=1e-9)
    assert history_title(plt.gcf()) == "R = 1.003"


def test_controller_ten_steps_lower_R(controller):
    net = reference_net(40)
    Xc, yc, gc = clusters(CENTERS, LABELS)
    sliders, buttons = controller(net, Xc, yc, gc)
    assert set(sliders) == set(NAMES13) | {"eta"}
    sliders["eta"].value = 0.5
    buttons["10 steps"].click()
    w = flat(net)
    # the same ten steps straight in torch
    ref = reference_net(40)
    for _ in range(10):
        R = F.binary_cross_entropy_with_logits(ref(Xc).reshape(-1), yc)
        R.backward()
        with torch.no_grad():
            for p in ref.parameters():
                p -= 0.5 * p.grad
                p.grad = None
    assert w == pytest.approx(flat(ref), abs=1e-5)
    R_after = oracle(net, Xc, yc)[0]
    assert R_after < 0.568
    assert history_title(plt.gcf()) == f"R = {R_after:.3f}"
    hist_ax = [ax for ax in plt.gcf().axes if ax.get_title().startswith("R = ")][0]
    assert len(hist_ax.lines[0].get_xdata()) == 11  # start plus ten steps
