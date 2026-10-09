"""The extra panels below the networks, chosen with explore(..., views=...).

"backprop": the backward pass of one case, step by step (see backprop.py). "cases": each case's slope along one
weight and their average. "slice": R over one or two weights with the path of
the steps. "line": R along the direction of the next step, with the first-order prediction. "hidden": the cases at
their hidden values with the output unit's boundary. "momentum": the next steps of gradient descent and of momentum.
Every function takes (ax, S, F) like the panels in draw.py.
"""

import numpy as np

from .draw import class_boundaries, class_shading, halo, letter, softmax

VIEWS = ("backprop", "cases", "slice", "line", "hidden", "momentum")


def message(ax, T, text):
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, color=T.muted)


def point_colours(S):
    """A colour per case: its class colour (classification) or a colour for its target y (regression)."""
    D, T = S.data, S.theme
    y = D.y.numpy()
    if D.task in ("binary", "multiclass"):
        return [T.class_colour[int(v)] for v in y]
    import matplotlib

    cmap = (
        matplotlib.colormaps[T.value_cmap] if hasattr(matplotlib, "colormaps") else matplotlib.cm.get_cmap(T.value_cmap)
    )
    span = float(y.max() - y.min()) or 1.0
    return [cmap((v - float(y.min())) / span) for v in y]


def moving_slopes(F):
    """The slopes of the weights that move; zero for the others."""
    g = np.zeros(len(F.w))
    idx = sorted(F.moving)
    g[idx] = F.g[idx]
    return g


# ---------------------------------------------------------------- each case's slope


def draw_cases(ax, S, F):
    """One strip per group: each case's slope of its own loss along the chosen weight; a short bar at the group mean;
    a dashed line at the average over all cases, which is the slope dR/dw of the table."""
    D, T = S.data, S.theme
    k = S.case_weight
    name = S.model.names[k]
    v = F.cases[:, k]
    colours = point_colours(S)
    for c in range(D.G):
        idx = np.nonzero(D.gsel[c])[0]
        m = len(idx)
        xs = c + (np.linspace(-0.22, 0.22, m) if m > 1 else np.zeros(1))
        order = np.argsort(v[idx], kind="stable")  # spread the dots along x in the order of their values
        ax.scatter(
            xs,
            v[idx][order],
            s=40 if m <= 8 else 10,
            c=[colours[i] for i in idx[order]],
            edgecolor=T.marker_edge,
            lw=0.3,
            zorder=3,
        )
        ax.hlines(float(v[idx].mean()), c - 0.32, c + 0.32, color=T.ink, lw=2, zorder=4)
    avg = float(v.mean())
    ax.axhline(0.0, color=T.zero_line, lw=0.8, zorder=0)
    ax.axhline(avg, ls="--", color=T.ink, lw=1.5, zorder=2, label=f"average {avg:+.4f} = dR/d{name}")
    ax.set_xticks(range(D.G))
    ax.set_xticklabels([letter(c) for c in range(D.G)])
    ax.set_xlim(-0.6, D.G - 0.4)
    ax.set_ylabel("slope of the case's own loss")
    formula = {"binary": "(p - y) da/dw", "multiclass": "Σk (pk - yk) dak/dw", "regression": "2(ŷ - y) dŷ/dw"}[D.task]
    title = f"each case's slope along {name}, {formula};\nbars: group means; dashed: the average"
    ax.set_title(title, fontsize=T.panel_title_size)
    ax.legend(fontsize=T.legend_size, loc="best")


# ---------------------------------------------------------------- R over one or two weights


def path_points(S, idx):
    """The steps taken, projected on the weights idx, with NaN rows where a new segment starts."""
    P = np.asarray(S.path, dtype=float)
    return P[:, idx]


def window(values, minimum=1.0):
    """A range around the given numbers with a margin."""
    values = np.asarray([v for v in np.ravel(values) if np.isfinite(v)])
    lo, hi = float(values.min()), float(values.max())
    half = max(minimum, 0.65 * (hi - lo))
    centre = 0.5 * (lo + hi)
    return centre - half, centre + half


def draw_slice(ax, S, F):
    """R over the slice weights with the other weights at their current values: contours (two weights) or a curve
    with its tangent (one weight); the start, the path of the steps, the current point and the direction of -grad R.
    With the momentum view on, the next steps of both methods as well."""
    D, M, T = S.data, S.model, S.theme
    idx = list(S.slice_weights)
    names = [M.names[k] for k in idx]
    w = np.asarray(F.w, dtype=float)
    w0 = np.asarray(S.w0, dtype=float)
    path = path_points(S, idx)
    others_move = bool(set(F.moving) - set(idx))
    note = "the other weights at their current values" if others_move or len(F.moving) != len(idx) else ""
    look = F.lookahead
    pts = [w[idx], w0[idx], path]
    if look:
        pts += [look["gd"]["path"][:, idx], look["momentum"]["path"][:, idx]]
    if len(idx) == 1:
        a = idx[0]
        lo, hi = window(np.concatenate([np.ravel(p) for p in pts]))
        u = np.linspace(lo, hi, 161)
        W = np.repeat(w[None, :], len(u), axis=0)
        W[:, a] = u
        R = M.risk_batch(D.X, D.y, W)
        ax.plot(u, R, color=T.ink, lw=1.8, label="R")
        ax.plot(
            u, F.R + F.g[a] * (u - w[a]), "--", color=T.prediction_colour, lw=1.2, label=f"tangent, slope {F.g[a]:+.3f}"
        )
        ax.plot([w[a]], [F.R], "o", color=T.ink, zorder=5)
        if w0[a] != w[a]:
            ax.axvline(w0[a], color=T.start_colour, ls=":", lw=1, label="start")
        span = float(R.max() - R.min()) or 1.0
        ax.set_ylim(float(R.min()) - 0.1 * span, float(R.max()) + 0.1 * span)
        ax.set_xlabel(names[0])
        ax.set_ylabel("R")
        ax.legend(fontsize=T.legend_size, loc="best")
        ax.set_title(f"R over {names[0]}" + (f",\n{note}" if note else ""), fontsize=T.panel_title_size)
        return
    a, b = idx
    flat = np.concatenate([p.reshape(-1, 2) for p in pts])
    xa, xb = window(flat[:, 0]), window(flat[:, 1])
    n = 41
    ua, ub = np.linspace(*xa, n), np.linspace(*xb, n)
    A, B = np.meshgrid(ua, ub)
    W = np.repeat(w[None, :], n * n, axis=0)
    W[:, a], W[:, b] = A.ravel(), B.ravel()
    R = M.risk_batch(D.X, D.y, W).reshape(n, n)
    levels = np.unique(np.quantile(R, [0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0]))
    if len(levels) > 1:
        ax.contourf(A, B, R, levels=levels, cmap=T.landscape_colormap(), alpha=0.35)
        ax.contour(A, B, R, levels=levels, colors=T.landscape_line, linewidths=0.6)
    if look:
        for key, colour, style, label in (
            ("gd", T.gd_colour, "o-", "next steps, gradient descent"),
            ("momentum", T.momentum_colour, "o--", f"next steps, momentum {S.beta:g}"),  # dashed: not by colour alone
        ):
            p = look[key]["path"][:, idx]
            ax.plot(p[:, 0], p[:, 1], style, color=colour, ms=2.5, lw=1.2, label=label, zorder=3)
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], "o-", color=T.path_colour, ms=3, lw=1.2, label="steps taken", zorder=4)
    ax.plot([w0[a]], [w0[b]], "o", mfc=T.paper, mec=T.marker_edge, ms=7, label="start", zorder=5)
    ax.plot([w[a]], [w[b]], "o", color=T.ink, ms=6, zorder=6)
    g = np.array([F.g[a], F.g[b]])
    norm = float(np.hypot(*g))
    if norm > 0:
        length = 0.25 * (xa[1] - xa[0])
        d = -g / norm * length
        ax.annotate(
            "",
            xy=(w[a] + d[0], w[b] + d[1]),
            xytext=(w[a], w[b]),
            arrowprops=dict(arrowstyle="-|>", color=T.ink, lw=1.8),
            zorder=7,
        )
    ax.set_xlim(*xa)
    ax.set_ylim(*xb)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.legend(fontsize=T.legend_size, loc="best")
    title = f"R over {names[0]} and {names[1]},\nwith the direction of -grad R"
    ax.set_title(title + (f"\n{note}" if note else ""), fontsize=T.panel_title_size)


# ---------------------------------------------------------------- R along the next step


def draw_line(ax, S, F):
    """R along the direction of the next step, w - t g (g: the slopes of the weights that move), for t from 0 to
    2.5 eta: the actual R, the tangent line R - t |g|^2 (the first-order prediction), where the step lands at
    t = eta on both, and the lowest point on this line."""
    D, M, T = S.data, S.model, S.theme
    g = moving_slopes(F)
    g2 = float((g**2).sum())
    if g2 == 0:
        message(ax, T, "every slope is 0:\nno step")
        return
    eta = F.eta
    ts = np.linspace(0, 2.5 * eta, 101)
    w = np.asarray(F.w, dtype=float)
    R = M.risk_batch(D.X, D.y, w[None, :] - ts[:, None] * g[None, :])
    R_eta = float(M.risk_batch(D.X, D.y, (w - eta * g)[None, :])[0])
    pred = F.R - eta * g2
    tangent = F.R - ts * g2
    ax.plot(ts, R, color=T.ink, lw=1.8, label="R along the step")
    ax.plot(ts, tangent, "--", color=T.prediction_colour, lw=1.2, label="first-order prediction")
    ax.axvline(eta, color=T.reference_line, lw=0.8, zorder=0)
    ax.plot([eta], [R_eta], "o", color=T.ink, ms=6, zorder=5, label=f"step of eta: R {R_eta:.4f}")
    ax.plot([eta], [pred], "o", mfc=T.paper, mec=T.prediction_colour, ms=7, zorder=5, label=f"predicted: {pred:.4f}")
    best = int(np.argmin(R))
    if 0 < best < len(ts) - 1:  # a minimum inside the range shown, not R still falling at its end
        ax.plot(
            [ts[best]],
            [R[best]],
            "v",
            color=T.momentum_colour,
            ms=8,
            zorder=5,
            label=f"lowest on the line: t = {ts[best]:.3g}",
        )
    top = max(float(R.max()), F.R)
    bottom = min(float(R.min()), pred)
    span = (top - bottom) or 1.0
    ax.set_ylim(bottom - 0.08 * span, top + 0.08 * span)
    ax.set_xlim(0, ts[-1])
    ax.set_xlabel("t, the step size along -grad R")
    ax.set_ylabel("R")
    ax.legend(fontsize=T.legend_size, loc="best")
    ax.set_title(f"R along the next step, eta {eta:g}", fontsize=T.panel_title_size)


# ---------------------------------------------------------------- the hidden layer


def hidden_width(S):
    return S.model.hidden[-1] if S.model.hidden else 0


def needs_3d(S):
    return "hidden" in S.views and hidden_width(S) == 3


def output_weights(S, w):
    """The output unit's weights for the last hidden layer and its bias."""
    k = S.model.layer_indices(len(S.model.layers) - 1)
    v = np.asarray([w[i] for i in k], dtype=float)
    return v[:-1], float(v[-1])


def output_layer(S, w):
    """The output layer as a matrix (one row per class, one column per unit of the last hidden layer) and its biases."""
    k = S.model.layer_indices(len(S.model.layers) - 1)
    V = np.asarray([w[i] for i in k], dtype=float).reshape(S.model.n_outputs, -1)
    return V[:, :-1], V[:, -1]


def draw_hidden_classes(ax, S, F, H, colours, small, label):
    """draw_hidden for K classes. One unit: p of each class against h, each case on the curve of its class. Two
    units: the most likely class over the (h1, h2) plane, with the boundaries. Three: the cases in 3-D; more: along
    the two directions in which h varies most; both coloured by class, without boundaries."""
    D, T = S.data, S.theme
    m = H.shape[1]
    V, b = output_layer(S, F.w)
    if m == 1:
        lo, hi = window(H[:, 0], minimum=0.6)
        u = np.linspace(lo, hi, 101)
        P = softmax(u[:, None] * V[None, :, 0] + b)
        for k in range(D.n_classes):
            ax.plot(u, P[:, k], color=T.class_colour[k], lw=1.5, label=f"p{k}")
        own = F.out.p[np.arange(D.n), D.y.numpy().astype(np.int64)]
        ax.scatter(H[:, 0], own, c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel("p of each class; each case on its own")
        ax.legend(fontsize=T.legend_size, loc="best")
        what = "p of each class"
    elif m == 2:
        x0, x1 = window(H[:, 0], minimum=0.6), window(H[:, 1], minimum=0.6)
        U1, U2 = np.meshgrid(np.linspace(*x0, 81), np.linspace(*x1, 81))
        A = U1[..., None] * V[:, 0] + U2[..., None] * V[:, 1] + b
        ax.imshow(class_shading(T, A), extent=(*x0, *x1), origin="lower", aspect="auto", interpolation="bilinear")
        class_boundaries(ax, U1, U2, A, colors=T.boundary)
        ax.scatter(H[:, 0], H[:, 1], c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        gh = [D.stats(H, c)[0] for c in range(D.G)]
        for c in range(D.G):
            ax.text(
                gh[c][0],
                gh[c][1],
                f"  {letter(c)}",
                fontsize=T.letter_size,
                zorder=4,
                path_effects=halo(T),
                color=T.ink,
            )
        ax.set_xlim(*x0)
        ax.set_ylim(*x1)
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel(f"{label}2")
        what = "the most likely class"
    elif m == 3 and hasattr(ax, "set_zlim"):
        ax.scatter(
            H[:, 0], H[:, 1], H[:, 2], c=colours, s=small, edgecolor=T.marker_edge, linewidths=0.3, depthshade=False
        )
        for q in range(3):
            getattr(ax, f"set_{'xyz'[q]}lim")(*window(H[:, q], minimum=0.6))
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel(f"{label}2")
        ax.set_zlabel(f"{label}3")
        ax.view_init(elev=20, azim=-60)
        what = "coloured by class"
    else:
        centred = H - H.mean(0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        two = centred @ vt[:2].T
        ax.scatter(two[:, 0], two[:, 1], c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        ax.set_xlabel("direction of h with the largest spread")
        ax.set_ylabel("the next one")
        what = "coloured by class"
    ax.set_title(f"the cases at their hidden values,\n{what}", fontsize=T.panel_title_size)


def draw_hidden(ax, S, F):
    """The cases at the values h of the last hidden layer. The output unit is a linear function of h: its boundary
    a = 0 is a point (one unit), a line (two) or a plane (three). With more units: the score a against the largest
    other direction of h."""
    D, T = S.data, S.theme
    m = hidden_width(S)
    if m == 0:
        message(ax, T, "no hidden layer:\nthe output unit sees x itself")
        return
    H = np.asarray(F.out.hidden[-1][1], dtype=float)
    colours = point_colours(S)
    small = 10 if D.n > 40 else 40
    label = "h" if len(S.model.hidden) == 1 else f"h{len(S.model.hidden)},"
    if D.task == "multiclass":
        draw_hidden_classes(ax, S, F, H, colours, small, label)
        return
    v, b = output_weights(S, F.w)
    binary = D.task == "binary"
    gh = [D.stats(H, c)[0] for c in range(D.G)]
    if m == 1:
        h = H[:, 0]
        lo, hi = window(h, minimum=0.6)
        u = np.linspace(lo, hi, 101)
        a = v[0] * u + b
        if binary:
            ax.plot(u, 1 / (1 + np.exp(-a)), color=T.neutral_colour, lw=1.5, label="p = sigmoid(a)")
            ax.scatter(h, D.y.numpy(), c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
            if v[0] != 0 and lo < -b / v[0] < hi:
                ax.axvline(-b / v[0], color=T.boundary, lw=1, label="boundary a = 0")
            ax.set_ylabel("p, and the class of each case")
        else:
            ax.plot(u, a, color=T.neutral_colour, lw=1.5, label="prediction")
            ax.scatter(h, D.y.numpy(), c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
            ax.set_ylabel("y")
        ax.set_xlabel(f"{label}1")
        ax.legend(fontsize=T.legend_size, loc="best")
    elif m == 2:
        x0, x1 = window(H[:, 0], minimum=0.6), window(H[:, 1], minimum=0.6)
        U1, U2 = np.meshgrid(np.linspace(*x0, 81), np.linspace(*x1, 81))
        a = v[0] * U1 + v[1] * U2 + b
        if binary:
            P = 1 / (1 + np.exp(-a))
            ax.contourf(U1, U2, P, levels=np.linspace(0, 1, 11), cmap=T.plane_colormap(), alpha=T.plane_alpha)
            if P.min() < 0.5 < P.max():
                ax.contour(U1, U2, P, levels=[0.5], colors=T.boundary)
        else:
            ax.contour(U1, U2, a, levels=10, cmap=T.value_cmap, linewidths=0.8)
        ax.scatter(H[:, 0], H[:, 1], c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        for c in range(D.G):
            ax.text(
                gh[c][0],
                gh[c][1],
                f"  {letter(c)}",
                fontsize=T.letter_size,
                zorder=4,
                path_effects=halo(T),
                color=T.ink,
            )
        ax.set_xlim(*x0)
        ax.set_ylim(*x1)
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel(f"{label}2")
    elif m == 3 and hasattr(ax, "set_zlim"):
        ax.scatter(
            H[:, 0], H[:, 1], H[:, 2], c=colours, s=small, edgecolor=T.marker_edge, linewidths=0.3, depthshade=False
        )
        r = [window(H[:, i], minimum=0.6) for i in range(3)]
        if binary and np.any(v != 0):
            i = int(np.argmax(np.abs(v)))  # solve the plane for the coordinate with the largest weight
            j, k = [q for q in range(3) if q != i]
            Gj, Gk = np.meshgrid(np.linspace(*r[j], 6), np.linspace(*r[k], 6))
            Gi = np.clip(-(v[j] * Gj + v[k] * Gk + b) / v[i], *r[i])
            coords = [None, None, None]
            coords[i], coords[j], coords[k] = Gi, Gj, Gk
            ax.plot_surface(*coords, color=T.plane_surface, alpha=0.25, linewidth=0)
        for q in range(3):
            getattr(ax, f"set_{'xyz'[q]}lim")(*r[q])
        ax.set_xlabel(f"{label}1")
        ax.set_ylabel(f"{label}2")
        ax.set_zlabel(f"{label}3")
        ax.view_init(elev=20, azim=-60)
    else:
        a = H @ v + b
        unit = v / (np.linalg.norm(v) or 1.0)
        rest = H - np.outer(H @ unit, unit)
        rest = rest - rest.mean(0)
        _, _, vt = np.linalg.svd(rest, full_matrices=False)
        other = rest @ vt[0]
        ax.scatter(a, other, c=colours, s=small, edgecolor=T.marker_edge, lw=0.3, zorder=3)
        if binary:
            ax.axvline(0.0, color=T.boundary, lw=1)
        ax.set_xlabel("a = output unit's score" if binary else "prediction")
        ax.set_ylabel("largest other direction of h")
    what = "the boundary a = 0" if binary else "coloured by y"
    ax.set_title(f"the cases at their hidden values,\n{what}", fontsize=T.panel_title_size)


# ---------------------------------------------------------------- gradient descent and momentum


def draw_momentum(ax, S, F):
    """R over the next steps from the current weights: gradient descent and heavy-ball momentum, where each step
    keeps the share beta of the last one."""
    T = S.theme
    look = F.lookahead
    gd, mo = look["gd"]["R"], look["momentum"]["R"]
    steps = range(len(gd))
    ax.plot(steps, gd, color=T.gd_colour, lw=1.8, label=f"gradient descent: R {gd[-1]:.4f}")
    ax.plot(steps, mo, "--", color=T.momentum_colour, lw=1.8, label=f"momentum, beta {S.beta:g}: R {mo[-1]:.4f}")
    if S.log_scale and min(min(gd), min(mo)) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("steps from the current weights")
    ax.set_ylabel("R")
    ax.legend(fontsize=T.legend_size, loc="best")
    ax.set_title(f"the next {len(gd) - 1} steps, eta {F.eta:g}", fontsize=T.panel_title_size)


VIEW_DRAWERS = {
    "cases": draw_cases,
    "slice": draw_slice,
    "line": draw_line,
    "hidden": draw_hidden,
    "momentum": draw_momentum,
}
