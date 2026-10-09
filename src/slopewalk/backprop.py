"""Backpropagation step by step: the stages of the backward pass, its numbers for every case, and how they are drawn.

The stages go from the output back to the first layer: delta at the output (dl/da = p - y, or 2(y-hat - y) for
regression); the slopes of the output weights (delta times the input of each line, 1 for the bias); then for each
hidden layer, from the last to the first: dl/dh (the sum over the next units of weight times delta), delta through
the activation (its derivative at z times dl/dh), and the slopes of that layer's weights. The last stage: R is the
average of the losses, so dR/dw is the average of the cases' slopes, the column of the table.
"""

from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib.font_manager import FontProperties

from .draw import get_renderer, letter, num, width_share
from .layout import PANEL_GAP

Stage = namedtuple("Stage", "kind layer text")  # kind: forward, output, weights, to_h, delta, average
Backward = namedtuple("Backward", "inputs z out delta dh deriv grads")  # per case, as numpy arrays


def layer_title(model, l):
    if l == len(model.layers) - 1:
        return "the output weights"
    if len(model.hidden) == 1:
        return "the first-layer weights"
    return f"the weights into layer {l + 1}"


def act_name(model, l):
    act = model.layers[l].act
    return act[0] if act else "no activation"


def backprop_stages(model):
    """The stages of the backward pass for this network, from the forward pass to the average over the cases."""
    L = len(model.layers)
    regression = model.task == "regression"
    output = {
        "binary": "at the output: δa = ∂ℓ/∂a = p - y",
        "multiclass": "at the output: δak = ∂ℓ/∂ak = pk - yk (yk is 1 for the case's class, else 0)",
        "regression": "at the output: δa = ∂ℓ/∂a = 2(ŷ - y)",
    }[model.task]
    out = [
        Stage("forward", None, "forward pass: z, h and " + ("ŷ" if regression else "a and p") + " of every case"),
        Stage("output", L - 1, output),
        Stage("weights", L - 1, f"slopes of {layer_title(model, L - 1)}: δ of the unit × the input of the line"),
    ]
    for l in range(L - 2, -1, -1):
        h = "h" if len(model.hidden) == 1 else f"h{l + 1},"
        act = act_name(model, l)
        out += [
            Stage("to_h", l, f"back to {h}: ∂ℓ/∂h = sum over the next units of weight × δ"),
            Stage("delta", l, f"through {act}: δ = {act}′(z) × ∂ℓ/∂h"),
            Stage("weights", l, f"slopes of {layer_title(model, l)}: δ of the unit × the input of the line"),
        ]
    out.append(Stage("average", None, "R is the average loss: dR/dw = the average of the cases' slopes (the table)"))
    return out


def derivative(layer, z, out):
    """The derivative of the layer's activation at z (out: its value there), elementwise."""
    m = layer.module
    if m is None or isinstance(m, nn.Identity):
        return torch.ones_like(z)
    if isinstance(m, nn.Tanh):
        return 1 - out**2
    if isinstance(m, nn.Sigmoid):
        return out * (1 - out)
    if isinstance(m, nn.ReLU):
        return (z > 0).to(z.dtype)
    raise ValueError(f"slopewalk: no derivative for {m}")


def backward_pass(model, X, y, w=None):
    """The forward and backward values of every case, with the formulas of the stages (no autograd).

    inputs[l]: the inputs of layer l (x, then the h of the layer before); z[l] and out[l]: its pre-activations and
    values; delta[l] = dl/dz[l] (the last layer: dl/da); dh[l] = dl/dout[l] for the hidden layers; deriv[l]: the
    activation's derivative at z[l]; grads: each case's slope along every weight, n x n_weights, in the order of the
    names. The network is left at w.
    """
    if w is not None:
        model.set_weights(w)
    layers = model.layers
    n = X.shape[0]
    with torch.no_grad():
        inputs, zs, outs = [], [], []
        h = X
        for layer in layers:
            inputs.append(h)
            z = layer.linear(h)
            h = layer.module(z) if layer.module is not None else z
            zs.append(z)
            outs.append(h)
        a = zs[-1]
        if model.task == "multiclass":  # p - y for every class, with y one-hot
            d_out = torch.softmax(a, dim=1) - nn.functional.one_hot(y.long(), a.shape[1]).to(a.dtype)
        elif model.task == "binary":
            d_out = torch.sigmoid(a) - y[:, None]
        else:
            d_out = 2 * (a - y[:, None])
        deltas = [None] * len(layers)
        dh = [None] * len(layers)
        deriv = [None] * len(layers)
        deltas[-1] = d_out
        for l in range(len(layers) - 2, -1, -1):
            dh[l] = deltas[l + 1] @ layers[l + 1].linear.weight
            deriv[l] = derivative(layers[l], zs[l], outs[l])
            deltas[l] = deriv[l] * dh[l]
        blocks = []
        for h_in, d in zip(inputs, deltas):
            with_one = torch.cat([h_in, torch.ones(n, 1, dtype=h_in.dtype)], dim=1)
            blocks.append((d[:, :, None] * with_one[:, None, :]).reshape(n, -1))
        grads = torch.cat(blocks, dim=1)

    def arr(ts):
        return [None if t is None else t.detach().numpy().astype(float) for t in ts]

    return Backward(arr(inputs), arr(zs), arr(outs), arr(deltas), arr(dh), arr(deriv), grads.numpy().astype(float))


def traced_case(S, c):
    """The case whose backward pass the derivation shows for group c: the point nearest the group's mean."""
    D = S.data
    idx = np.nonzero(D.gsel[c])[0]
    X = D.X.numpy()[idx]
    return int(idx[np.argmin(((X - X.mean(0)) ** 2).sum(1))])


# ---------------------------------------------------------------- the network panels


def stage_index(S):
    return max(0, min(S.backprop_step, len(S.backprop_stages) - 1))


def line_styles(S, F):
    """Colour and opacity of each weight's line at the current stage, or None for the usual drawing."""
    st = F.stage
    if st is None or st.kind == "forward":
        return None
    M, T = S.model, S.theme
    styles = {k: (None, 0.25) for k in range(M.n_weights)}  # faint, in the weight's own colour
    if st.kind == "weights":
        for k in M.layer_indices(st.layer):
            styles[k] = (T.backprop_colour, 1.0)
    elif st.kind == "average":
        for k in styles:
            styles[k] = (T.backprop_colour, 1.0)
    elif st.kind == "to_h":
        for k in M.layer_indices(st.layer + 1):
            if M.where[k][2] >= 0:  # not the bias
                styles[k] = (None, 1.0)
    return styles


def shared_labels(F):
    """True when the labels on the lines are the same in every network panel (the weights themselves), so that the
    first panel alone carries them. The slopes of the backward pass differ from case to case and go in every panel."""
    return F.stage is None or F.stage.kind in ("forward", "to_h")


def label_spec(S, F, c):
    """The labels on the lines at the current stage: {k: (long text, short text, colour)}, or None for the usual
    labels. Weights stages: each line's slope (mean over the group); back to h: the weights used; else none."""
    st = F.stage
    if st is None or st.kind == "forward":
        return None
    M, T, D = S.model, S.theme, S.data
    g = F.bp.grads[D.gsel[c]].mean(0)
    spec = {}
    if st.kind in ("weights", "average"):
        ks = M.layer_indices(st.layer) if st.kind == "weights" else range(M.n_weights)
        for k in ks:
            long = f"∂ℓ/∂{M.names[k]} = {g[k]:+.3f}" if c == 0 else f"{g[k]:+.3f}"  # names in panel A only
            spec[k] = (long, f"{g[k]:+.3f}", T.backprop_colour, f"{g[k]:+.2f}")
    elif st.kind == "to_h":
        for k in M.layer_indices(st.layer + 1):
            if M.where[k][2] >= 0:
                colour = T.positive if F.w[k] >= 0 else T.negative_text
                spec[k] = (f"{M.names[k]} = {F.w[k]:.2f}", f"{F.w[k]:.2f}", colour)
    return spec


def box(ax, x, y, text, T):
    """A backward value under the forward box at (x, y), far enough down to clear it: both boxes have the same number
    of lines (two for single points, three with a standard deviation)."""
    lines = text.count("\n") + 1
    drop = lines * 1.2 * (T.value_size + T.backprop_size) / 2 + 0.25 * T.value_size + 0.2 * T.backprop_size + 3
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, -max(31.0, drop)),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=T.backprop_size,
        color=T.backprop_colour,
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", fc=T.backprop_tint, ec=T.backprop_colour, lw=0.8),
    )


def tangent(ax, xc, yc, act, m, T, height):
    """The tangent of the activation at m on its glyph: its slope is the derivative used at this stage."""
    name, f, lim, df = act
    w_, h_ = 0.62, height
    u = np.linspace(-lim, lim, 100)
    vals = f(u)
    lo, span = vals.min(), vals.max() - vals.min()
    m = float(np.clip(m, -lim, lim))
    ts = np.linspace(max(-lim, m - 0.35 * lim), min(lim, m + 0.35 * lim), 2)
    ys = f(m) + df(m) * (ts - m)

    def gx(t):
        return xc - w_ / 2 + (t + lim) / (2 * lim) * w_

    def gy(v):
        return yc + 0.85 * h_ * ((v - lo) / span - 0.5)

    ax.plot(gx(ts), gy(ys), color=T.backprop_colour, lw=1.6, zorder=5)


def overlay(ax, S, F, c, edges):
    """The backward values on network panel c, for every stage reached so far: delta at the output, dl/dh under the
    h boxes, delta under the z boxes; at the "through the activation" stage, the tangents on the glyphs; at the
    "back to h" stage, arrows back along the lines used."""
    st = F.stage
    if st is None or st.kind == "forward":
        return
    M, D, L, T = S.model, S.data, S.layout, S.theme
    bp = F.bp
    sel = D.gsel[c]
    reached = S.backprop_stages[1 : stage_index(S) + 1]
    tag = (lambda name: f"{name}\n") if c == 0 else (lambda name: "")  # the names once, in panel A
    for s in reached:
        if s.kind == "output":  # under each score: one box, or one per class
            for k, ty in enumerate(L.unit_ys(M.n_outputs)):
                m, sd = D.stats(bp.delta[-1][:, k], c)
                box(ax, L.x_a, ty, tag("δa" if M.n_outputs == 1 else f"δa{k}") + num(m, sd), T)
        elif s.kind in ("to_h", "delta"):
            l = s.layer
            XZ, XG, XH = L.blocks[l]
            tys = L.unit_ys(M.layers[l].out_features)
            values = bp.dh[l] if s.kind == "to_h" else bp.delta[l]
            for j, ty in enumerate(tys):
                m, sd = D.stats(values[:, j], c)
                nm = M.unit_name(l, j)
                if s.kind == "to_h":
                    box(ax, XH, ty, tag(f"∂ℓ/∂h{nm}") + num(m, sd), T)
                else:
                    box(ax, XZ, ty, tag(f"δ{nm}") + num(m, sd), T)
    if st.kind == "delta" and M.layers[st.layer].act:
        l = st.layer
        XZ, XG, XH = L.blocks[l]
        tys = L.unit_ys(M.layers[l].out_features)
        height = min(0.5, 0.75 * (tys[0] - tys[1])) if len(tys) > 1 else 0.5
        z = bp.z[l][sel].mean(0)
        for j, ty in enumerate(tys):
            tangent(ax, XG, ty, M.layers[l].act, z[j], T, height)
    if st.kind == "to_h":
        arrow = dict(arrowstyle="-|>", color=T.backprop_colour, lw=1.4, shrinkA=0, shrinkB=0)
        for k, p0, p1, _ in edges:
            if M.where[k][0] == st.layer + 1 and M.where[k][2] >= 0:
                a = (p0[0] + 0.62 * (p1[0] - p0[0]), p0[1] + 0.62 * (p1[1] - p0[1]))
                b = (p0[0] + 0.38 * (p1[0] - p0[0]), p0[1] + 0.38 * (p1[1] - p0[1]))
                ax.annotate("", xy=b, xytext=a, arrowprops=arrow, zorder=4)


# ---------------------------------------------------------------- the derivation panel


def factor(v, digits=3):
    """A number as the second factor of a product: in parentheses when it is negative."""
    text = f"{v:+.{digits}f}"
    return f"({text})" if v < 0 else text


def derivation_lines(S, F, i):
    """The steps of the backward pass for case i, as (stage index, text) pairs."""
    M = S.model
    bp = F.bp
    y = float(S.data.y[i])
    a = bp.z[-1][i, 0]
    regression = S.task == "regression"
    out = []
    for s_idx, st in enumerate(S.backprop_stages):
        if S.task == "multiclass" and st.kind in ("forward", "output"):
            out += [(s_idx, text) for text in class_lines(bp.z[-1][i], int(y), st.kind)]
        elif st.kind == "forward":
            if regression:
                out.append((s_idx, f"forward: ŷ = {a:+.3f}, y = {y:+.3f}"))
            else:
                p = 1 / (1 + np.exp(-a))
                out.append((s_idx, f"forward: a = {a:+.3f}, p = {p:.3f}, y = {y:g}"))
        elif st.kind == "output":
            d = bp.delta[-1][i, 0]
            if regression:
                out.append((s_idx, f"δa = 2(ŷ - y) = 2({a:+.3f} - {y:+.3f}) = {d:+.3f}"))
            else:
                p = 1 / (1 + np.exp(-a))
                out.append((s_idx, f"δa = p - y = {p:.3f} - {y:g} = {d:+.3f}"))
        elif st.kind == "weights":
            l = st.layer
            labels = M.input_labels(l)
            for k in M.layer_indices(l):
                _, j, src = M.where[k]
                unit = output_name(M, j) if l == len(M.layers) - 1 else M.unit_name(l, j)
                d = bp.delta[l][i, j]
                if src < 0:
                    out.append((s_idx, f"{M.names[k]}: δ{unit} × 1 = {bp.grads[i, k]:+.3f}"))
                else:
                    v = bp.inputs[l][i, src]
                    out.append(
                        (
                            s_idx,
                            f"{M.names[k]}: δ{unit} × {labels[src]} = {d:+.3f} × {factor(v)} = {bp.grads[i, k]:+.3f}",
                        )
                    )
        elif st.kind == "to_h":
            l = st.layer
            nxt = M.layers[l + 1]
            Wn = nxt.linear.weight.detach().numpy()
            for j in range(M.layers[l].out_features):
                nm = M.unit_name(l, j)
                if nxt.out_features == 1:
                    k = M.layer_indices(l + 1)[j]
                    unit = output_name(M, 0) if l + 1 == len(M.layers) - 1 else M.unit_name(l + 1, 0)
                    d = bp.delta[l + 1][i, 0]
                    product = f"{Wn[0, j]:+.3f} × {factor(d)} = {bp.dh[l][i, j]:+.3f}"
                    out.append((s_idx, f"∂ℓ/∂h{nm} = {M.names[k]} × δ{unit} = {product}"))
                else:  # a sum over the next units, two terms to a line so that the lines fit their column
                    terms = [f"{Wn[q, j]:+.2f} × {factor(bp.delta[l + 1][i, q])}" for q in range(nxt.out_features)]
                    parts = [" + ".join(terms[q : q + 2]) for q in range(0, len(terms), 2)]
                    parts[0] = f"∂ℓ/∂h{nm} = {parts[0]}"
                    parts[1:] = [f"      + {part}" for part in parts[1:]]
                    parts[-1] += f" = {bp.dh[l][i, j]:+.3f}"
                    out += [(s_idx, part) for part in parts]
        elif st.kind == "delta":
            l = st.layer
            act = act_name(S.model, l)
            for j in range(M.layers[l].out_features):
                nm = M.unit_name(l, j)
                out.append(
                    (
                        s_idx,
                        f"δ{nm} = {act}′(z{nm}) × ∂ℓ/∂h{nm} = {bp.deriv[l][i, j]:.3f} × {factor(bp.dh[l][i, j])} "
                        f"= {bp.delta[l][i, j]:+.3f}",
                    )
                )
        elif st.kind == "average":
            k = M.layer_indices(len(M.layers) - 1)[0]
            mean = bp.grads[:, k].mean()
            out.append((s_idx, f"dR/d{M.names[k]} = mean over the {S.data.n} cases = {mean:+.4f}"))
    return out


def output_name(model, j):
    """The name of output unit j: a with one output, a0, a1, ... with one per class."""
    return "a" if model.n_outputs == 1 else f"a{j}"


def class_lines(a, y, kind):
    """The lines of the forward and output stages for K classes: the scores a of one case and its class y."""
    p = np.exp(a - a.max())
    p /= p.sum()
    if kind == "forward":
        scores = ", ".join(f"{v:+.3f}" for v in a)
        probabilities = ", ".join(f"{v:.3f}" for v in p)
        return [f"forward: a = ({scores}), y = {y}", f"p = softmax(a) = ({probabilities})"]
    return [
        f"δa{k} = p{k} - {int(k == y)} = {p[k]:.3f} - {int(k == y)} = {p[k] - (k == y):+.3f}" for k in range(len(a))
    ]


def fitting_size(ax, T, texts, columns, floor=9.5):
    """The largest font from T.note_size down to floor (in half points) at which the longest text, in bold, fits its
    column of the panel. Measured in inches from the panel's share of the figure, so the same in every frame."""
    renderer = get_renderer(ax.figure)
    if renderer is None or not texts:
        return T.note_size
    column = (ax.figure.get_figwidth() * width_share(ax) - PANEL_GAP) / columns * 0.96
    size = T.note_size
    while size > floor:
        props = FontProperties(size=size, weight="bold")
        longest = max(renderer.get_text_width_height_descent(t, props, ismath=False)[0] for t in texts)
        if longest / ax.figure.dpi <= column:
            break
        size -= 0.5
    return size


def draw_derivation(ax, S, F):
    """The Worked example: the backward pass of one case, step by step; the stages reached so far, the current one in
    bold. Its heading, "Worked example" in the colour of the backward pass, sits above the title of the stage."""
    T = S.theme
    ax.axis("off")
    c = max(0, min(S.backprop_case, S.data.G - 1))
    i = traced_case(S, c)
    step = stage_index(S)
    stage = S.backprop_stages[step]
    lines = [(s, t) for s, t in derivation_lines(S, F, i) if s <= step]
    per_column, columns = 12, 2  # the steps run down the first column, then the second
    most = per_column * columns
    shown = lines[-most:]
    if S.data.gsel[c].sum() == 1:
        case = f": case {letter(c)}"
    else:
        case = f": the case nearest the mean of group {letter(c)}"
    title = ax.set_title(
        f"stage {step} of {len(S.backprop_stages) - 1}: {stage.text}", fontsize=T.panel_title_size, loc="left"
    )
    # the heading above the title: "Worked example" in the colour of the backward pass, then the case in ink
    head = ax.annotate(
        "Worked example",
        xy=(0, 1),
        xycoords=title,
        xytext=(0, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=T.panel_title_size,
        fontweight="bold",
        color=T.backprop_colour,
    )
    ax.annotate(case, xy=(1, 0), xycoords=head, ha="left", va="bottom", fontsize=T.panel_title_size, color=T.ink)
    dy = 0.96 / per_column
    if len(lines) > most:
        shown = [(None, "... (earlier steps above)")] + shown[1:]
    fontsize = fitting_size(ax, T, [t for _, t in derivation_lines(S, F, i)], columns)
    for n, (s, text) in enumerate(shown):
        current = s == step
        col, row = divmod(n, per_column)
        ax.text(
            0.5 * col,
            0.98 - row * dy,
            text,
            transform=ax.transAxes,
            fontsize=fontsize,
            va="top",
            color=T.backprop_colour if current else (T.derivation_muted if s is None else T.derivation_text),
            fontweight="bold" if current else "normal",
        )
