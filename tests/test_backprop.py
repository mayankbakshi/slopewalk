"""The backward pass step by step: its numbers against autograd, its stages, its drawing, widgets and recording."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from helpers import NETS, make, points, regression_points
from render_examples import reference_net

from slopewalk import Session, explore, xor_centres
from slopewalk.backprop import backprop_stages, backward_pass, derivation_lines, traced_case
from slopewalk.draw import THEME, width_share
from slopewalk.model import Model

XOR = xor_centres()


@pytest.fixture
def static(monkeypatch):
    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def autograd_values(model, X, y):
    """dl_i/dz and dl_i/dh for every layer and case, from autograd: the oracle for the backward pass."""
    zs, hs = [], []
    h = X
    for L in model.layers:
        z = F.linear(h, L.linear.weight.detach(), L.linear.bias.detach())
        if not z.requires_grad:
            z.requires_grad_(True)
        h = L.module(z) if L.module is not None else z
        zs.append(z)
        hs.append(h)
    losses = model.loss(h.reshape(-1), y, reduction="none")
    grads = torch.autograd.grad(losses.sum(), zs + hs[:-1])
    return [g.numpy() for g in grads[: len(zs)]], [g.numpy() for g in grads[len(zs) :]]


@pytest.mark.parametrize("task", ["binary", "regression"])
@pytest.mark.parametrize("name", [n for n in NETS if n != "wide"])
def test_backward_pass_matches_autograd(name, task):
    net = make(name)
    X, y = points(net, n=9) if task == "binary" else regression_points(net)
    M = Model(net, task=task)
    bp = backward_pass(M, X, y)
    dz, dh = autograd_values(M, X, y)
    for l in range(len(M.layers)):
        assert bp.delta[l] == pytest.approx(dz[l], abs=1e-6)
    for l in range(len(M.layers) - 1):
        assert bp.dh[l] == pytest.approx(dh[l], abs=1e-6)
    assert bp.grads == pytest.approx(M.case_slopes(X, y), abs=1e-6)
    assert bp.grads.mean(0) == pytest.approx(M.slopes(X, y), abs=1e-6)  # the average is dR/dw


def test_relu_derivative_at_zero_matches_autograd():
    net = make("relu4")
    with torch.no_grad():
        net[0].bias[0] = 0.0
        net[0].weight[0] = 0.0  # z1 = 0 for every case
    X, y = points(net)
    M = Model(net)
    bp = backward_pass(M, X, y)
    dz, _ = autograd_values(M, X, y)
    assert bp.delta[0] == pytest.approx(dz[0], abs=1e-7) and np.all(bp.deriv[0][:, 0] == 0)


def test_stages():
    kinds = [(s.kind, s.layer) for s in backprop_stages(Model(make("tanh3")))]
    assert kinds == [
        ("forward", None),
        ("output", 1),
        ("weights", 1),
        ("to_h", 0),
        ("delta", 0),
        ("weights", 0),
        ("average", None),
    ]
    assert len(backprop_stages(Model(make("logistic")))) == 4
    assert len(backprop_stages(Model(make("two_hidden")))) == 10
    texts = [s.text for s in backprop_stages(Model(make("tanh3")))]
    assert "p - y" in texts[1] and "through tanh" in texts[4]
    assert "2(ŷ - y)" in backprop_stages(Model(make("tanh3"), task="regression"))[1].text


def test_derivation_multiplies_out():
    s = Session(reference_net(40), *XOR, views=("backprop",))
    M = s.model
    bp = backward_pass(M, s.data.X, s.data.y)
    lines = dict(derivation_lines(s, type("F", (), {"bp": bp})(), 0))
    text = "\n".join(t for _, t in derivation_lines(s, type("F", (), {"bp": bp})(), 0))
    a = bp.z[-1][0, 0]
    p = 1 / (1 + np.exp(-a))
    assert f"δa = p - y = {p:.3f} - 1 = {p - 1:+.3f}" in text
    assert f"w10: δa × h1 = {p - 1:+.3f} × {bp.inputs[1][0, 0]:+.3f} = {bp.grads[0, 9]:+.3f}" in text
    assert f"= {bp.delta[0][0, 0]:+.3f}" in text and "tanh′(z1)" in text
    assert lines  # every stage has at least one line
    stages_with_lines = {st for st, _ in derivation_lines(s, type("F", (), {"bp": bp})(), 0)}
    assert stages_with_lines == set(range(len(s.backprop_stages)))


def test_traced_case_is_nearest_the_group_mean():
    s = Session(make("tanh3"), *xor_centres(noise=0.35, n_per=20))
    i = traced_case(s, 1)
    X = s.data.X.numpy()
    idx = np.nonzero(s.data.gsel[1])[0]
    d = ((X[idx] - X[idx].mean(0)) ** 2).sum(1)
    assert i == idx[np.argmin(d)]


def backward_boxes(ax, T=THEME):
    return [t for t in ax.texts if t.get_fontsize() == T.backprop_size and t.get_color() == T.backprop_colour]


def network_panels(fig):
    return [ax for ax in fig.axes if re.match(r"^[A-Z]: ", ax.get_title())]


@pytest.mark.parametrize("numbers", ["few", "all"])
def test_pictures_of_every_stage(numbers, static):
    """numbers="all": every stage reached so far, in every panel (delta_a, then dl/dh1..3, then delta1..3). "few",
    the default: only the boxes the current stage works with, in the Worked example's panel (A here)."""
    s = explore(reference_net(40), *XOR, views="backprop", numbers=numbers)
    expected_boxes = [0, 1, 1, 4, 7, 7, 7] if numbers == "all" else [0, 1, 1, 4, 6, 3, 0]
    for k, n_boxes in enumerate(expected_boxes):
        s.backprop_step = k
        fig = s.draw()
        panels = network_panels(fig)
        assert len(panels) == 4
        others = n_boxes if numbers == "all" else 0
        assert [len(backward_boxes(ax)) for ax in panels] == [n_boxes] + [others] * 3
        labels = [t for t in panels[0].texts if t.get_fontsize() == THEME.edge_label_size]
        kind = s.backprop_stages[k].kind
        assert (
            len(labels)
            == {"forward": 13, "output": 0, "weights": 4 if k == 2 else 9, "to_h": 3, "delta": 0, "average": 13}[kind]
        )
        derivation = [ax for ax in fig.axes if heading(ax).startswith("Worked example: case A")]
        assert len(derivation) == 1 and f"stage {k} of 6" in derivation[0].get_title(loc="left")
        plt.close("all")


def heading(ax):
    """The heading of the Worked example panel, "Worked example" and its case, or "" for the other panels."""
    texts = [t.get_text() for t in ax.texts]
    return "Worked example" + texts[texts.index("Worked example") + 1] if "Worked example" in texts else ""


def test_worked_example_heading(static):
    """The heading: Worked example in the colour of the backward pass and in bold, the case after it in ink, the stage
    below."""
    s = explore(reference_net(40), *XOR, views="backprop", backprop_step=1)
    ax = [a for a in s.fig.axes if heading(a)][0]
    head = [t for t in ax.texts if t.get_text() == "Worked example"][0]
    case = ax.texts[ax.texts.index(head) + 1]
    assert head.get_color() == s.theme.backprop_colour and head.get_fontweight() == "bold"
    assert case.get_text() == ": case A" and case.get_color() == s.theme.ink
    assert ax.get_title(loc="left").startswith("stage 1 of 6: at the output")
    renderer = s.fig.canvas.get_renderer()
    above = head.get_window_extent(renderer)
    title = ax._left_title.get_window_extent(renderer)
    after = case.get_window_extent(renderer)
    assert above.y0 >= title.y1 and after.x0 >= above.x1 - 1 and abs(after.y0 - above.y0) < 1
    assert above.y1 <= s.fig.bbox.y1  # inside the figure


def test_slope_labels_are_the_group_means(static):
    s = explore(reference_net(40), *XOR, views="backprop", backprop_step=2)
    ax = network_panels(s.fig)[0]
    labels = sorted(t.get_text() for t in ax.texts if t.get_fontsize() == THEME.edge_label_size)
    bp = backward_pass(s.model, s.data.X, s.data.y)
    assert labels == sorted(f"{bp.grads[0, k]:+.3f}" for k in (9, 10, 11, 12))


def test_options(static):
    s = explore(reference_net(40), *XOR, backprop_step=3, backprop_case="C")
    assert "backprop" in s.views and s.backprop_step == 3 and s.backprop_case == 2
    assert any(heading(ax).startswith("Worked example: case C") for ax in s.fig.axes)
    with pytest.raises(ValueError, match="backprop_step must be a whole number from 0 to 6"):
        explore(reference_net(40), *XOR, backprop_step=7)
    with pytest.raises(ValueError, match="is not a group"):
        explore(reference_net(40), *XOR, backprop_case="Z")


def test_the_derivation_is_the_widest_lower_panel(static):
    for style, share in (("wide", 0.3), ("standard", 1.0)):
        s = explore(reference_net(40), *XOR, views="backprop", layout=style, backprop_step=1)
        ax = [a for a in s.fig.axes if heading(a).startswith("Worked example: case A")][0]
        others = [a for a in s.fig.axes[4:] if a is not ax]
        assert width_share(ax) >= share and all(width_share(a) < width_share(ax) for a in others)
        plt.close("all")


def test_record_backprop(tmp_path):
    from PIL import Image

    s = Session(reference_net(40), *XOR)
    w = s.weights()
    path = s.record_backprop(tmp_path / "bp.gif", dpi=30)
    with Image.open(path) as im:
        assert im.n_frames == 7
    assert s.weights() == w and s.views == () and s.backprop_step == 0
    assert plt.get_fignums() == []


def test_the_numbers_follow_the_worked_example(static):
    """By default (numbers="few") the backward numbers of the current stage are in the Worked example's panel, with
    the names of the values; the other panels are faded, their lines faint; at "back to h" they say whose weights
    are labelled; the Worked example lists the current stage only; the average stage is drawn in every panel."""
    s = explore(reference_net(40), *XOR, views="backprop", backprop_step=2, backprop_case="D")
    T = s.theme
    P = network_panels(s.fig)
    assert [len(backward_boxes(ax)) for ax in P] == [0, 0, 0, 1]
    assert any(t.get_text().startswith("δa\n") for t in P[3].texts)
    assert any(t.get_text().startswith("h1\n") for t in P[3].texts)  # the names moved to panel D
    assert not any(t.get_text().startswith(("h1\n", "δa")) for ax in P[:3] for t in ax.texts)
    labels = [t for t in P[3].texts if t.get_fontsize() == THEME.edge_label_size]
    assert [sum(t.get_fontsize() == THEME.edge_label_size for t in ax.texts) for ax in P] == [0, 0, 0, 4]
    assert all(t.get_color() == T.backprop_colour for t in labels)  # the slopes of case D (values: 13 weights)
    for ax in P[:3]:  # a quiet panel: every line thin and faint
        lines = [ln for ln in ax.lines if ln.get_zorder() == 1]
        assert all(ln.get_alpha() == T.faded_line_alpha and ln.get_linewidth() == T.line_width for ln in lines)
    assert sum(ln.get_alpha() == 1.0 for ln in P[3].lines if ln.get_zorder() == 1) == 4
    worked = [ax for ax in s.fig.axes if heading(ax).startswith("Worked example: case D")][0]
    lines = [t.get_text() for t in worked.texts if t.get_text().startswith(("w1", "δ", "forward", "p ="))]
    assert len(lines) == 4 and all(t.startswith("w1") for t in lines)  # the current stage only: w10 to w13
    s.backprop_step = 3  # back to h: the weights used, labelled in panel D, the others marked
    P = network_panels(s.draw())
    assert [sum(t.get_fontsize() == THEME.edge_label_size for t in ax.texts) for ax in P] == [0, 0, 0, 3]
    assert all(any(t.get_text() == "weights as in D" for t in ax.texts) for ax in P[:3])
    s.backprop_step = 6  # the average over the cases: every panel's slopes, the names in panel A
    P = network_panels(s.draw())
    assert [sum(t.get_fontsize() == THEME.edge_label_size for t in ax.texts) for ax in P] == [13, 13, 13, 13]
    assert any(t.get_text().startswith("h1\n") for t in P[0].texts)  # the names back in panel A
    s.backprop_step = 4
    worked = [ax for ax in s.draw().axes if heading(ax).startswith("Worked example: case D")][0]
    assert sum(t.get_text().startswith("δ") for t in worked.texts) == 3  # δ1 to δ3, nothing from earlier stages
    with pytest.raises(ValueError, match="numbers must be 'few' or 'all'"):
        Session(reference_net(40), *XOR, numbers="some")
    plt.close("all")


def test_widget_controls(monkeypatch):
    pytest.importorskip("ipywidgets")
    import IPython.display

    monkeypatch.delenv("NB_STATIC", raising=False)
    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: None)
    s = explore(reference_net(40), *XOR, views="backprop")
    C = s.controller
    assert C.tabs.selected_index == 1 and "backprop" in s.views and "Press next stage" in C.message.value
    png = C.image.value
    C.bp_next.click()
    C.bp_next.click()
    assert s.backprop_step == 2 and C.image.value != png
    assert C.message.value.count("In panel A: on the purple lines") == 1  # the numbers are in panel A
    assert not C.bp_prev.disabled and not C.bp_next.disabled
    C.bp_prev.click()
    assert s.backprop_step == 1
    C.bp_case.value = "D"
    assert s.backprop_case == 3 and any(heading(ax).startswith("Worked example: case D") for ax in s.fig.axes)
    assert "In panel D: under a, the purple box is δa = p - y of the case." in C.message.value
    boxes = [len(backward_boxes(ax, s.theme)) for ax in network_panels(s.fig)]  # the wide layout scales the text
    assert boxes == [0, 0, 0, 1]  # the numbers moved with the case
    png = C.image.value
    C.tabs.selected_index = 0  # the other tab: the backward pass stays in the picture, which does not change
    assert "backprop" in s.views and s.backprop_step == 1 and C.image.value == png
    C.bp_prev.click()
    assert s.backprop_step == 0 and C.bp_prev.disabled and not C.bp_next.disabled  # the buttons stop at the ends
    for _ in range(6):
        C.bp_next.click()
    assert s.backprop_step == 6 and C.bp_next.disabled and not C.bp_prev.disabled
    C.to_stage(+1)  # nothing beyond the last stage
    assert s.backprop_step == 6
    assert not hasattr(C, "bp_step")  # no stage slider: the picture names the stage
    starts_on = explore(reference_net(40), *XOR, backprop_step=3).controller
    assert starts_on.tabs.selected_index == 1
    plt.close("all")


def test_shared_labels_only_in_the_first_panel(static):
    """Weights (forward, back to h) are the same in every panel: panel A only. Per-case slopes: every panel with
    numbers="all"; by default only the Worked example's panel, except at the average stage, which is about every
    case."""
    count = lambda fig: [sum(t.get_fontsize() == THEME.edge_label_size for t in ax.texts) for ax in network_panels(fig)]  # noqa: E731
    s = explore(reference_net(40), *XOR, views="backprop", numbers="all")
    for k, expected in ((0, [13, 0, 0, 0]), (2, [4, 4, 4, 4]), (3, [3, 0, 0, 0]), (6, [13, 13, 13, 13])):
        s.backprop_step = k
        assert count(s.draw()) == expected, k
        plt.close("all")
    few = explore(reference_net(40), *XOR, views="backprop")
    for k, expected in ((0, [13, 0, 0, 0]), (2, [4, 0, 0, 0]), (3, [3, 0, 0, 0]), (6, [13, 13, 13, 13])):
        few.backprop_step = k
        assert count(few.draw()) == expected, k
        plt.close("all")


def test_names_once_in_the_backpropagation_view(static):
    """Names that repeat in every panel (z, h, a, p, δ, ∂ℓ/∂h, and the slopes' names) are written in panel A only;
    the other panels show the values."""
    for numbers in ("all", "few"):
        s = explore(make("logistic"), *XOR, views="backprop", backprop_step=2, numbers=numbers)  # 3 weights: names
        P = network_panels(s.fig)
        first = [t.get_text() for t in P[0].texts]
        assert any(t.startswith("a\n") for t in first) and any(t.startswith("δa\n") for t in first)
        assert any(t.startswith("∂ℓ/∂w1 = ") for t in first)
        for ax in P[1:]:
            texts = [t.get_text() for t in ax.texts]
            assert not any(t.startswith(("a\n", "p\n", "δa\n", "∂ℓ/∂")) for t in texts)
            # "all": the same boxes, values only; "few": no backward numbers outside the Worked example's panel
            assert len(backward_boxes(ax)) == (len(backward_boxes(P[0])) if numbers == "all" else 0)
        plt.close("all")
    plain = explore(make("logistic"), *XOR)  # outside the backpropagation view every panel keeps its names
    assert all(any(t.get_text().startswith("a\n") for t in ax.texts) for ax in network_panels(plain.fig))
    plt.close("all")


def test_each_stage_fades_what_it_does_not_use(static):
    """Through tanh: z, the curves, h, dl/dh and the new deltas stay; a and delta a fade (with numbers="all"; by
    default the delta a boxes of the earlier stage are not drawn at all). Slopes of the output weights: the h boxes
    and delta a stay, x and z fade, and the lines of the first layer are thin and faint. The forward pass fades
    nothing."""
    few = explore(reference_net(40), *XOR, views="backprop", backprop_step=4)
    assert not any(t.get_text().startswith("δa") for t in network_panels(few.fig)[0].texts)
    s = explore(reference_net(40), *XOR, views="backprop", backprop_step=4, numbers="all")
    ax = network_panels(s.fig)[0]

    def alpha(start):
        texts = [t for t in ax.texts if t.get_text().startswith(start)]
        assert texts, start
        return {t.get_alpha() for t in texts}

    T = s.theme
    assert alpha("z1") == {None} and alpha("h1") == {None} and alpha("δ1") == {None} and alpha("∂ℓ/∂h1") == {None}
    assert alpha("a\n") == {T.faded_text_alpha} and alpha("δa") == {T.faded_text_alpha}
    boxes = [t.get_bbox_patch().get_alpha() for t in ax.texts if t.get_text().startswith("a\n")]
    assert boxes == [T.faded_alpha]  # the box fades more than its number
    s.backprop_step = 2
    ax = network_panels(s.draw())[0]
    assert alpha("h1") == {None} and alpha("δa") == {None} and alpha("z1") == {T.faded_text_alpha}
    lines = [ln for ln in ax.lines if ln.get_zorder() == 1]
    first = s.model.layer_indices(0)
    assert all(lines[k].get_alpha() == T.faded_line_alpha and lines[k].get_linewidth() == T.line_width for k in first)
    assert all(lines[k].get_alpha() == 1.0 for k in s.model.layer_indices(1))
    s.backprop_step = 0
    ax = network_panels(s.draw())[0]
    assert all(t.get_alpha() is None for t in ax.texts)
