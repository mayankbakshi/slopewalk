"""Networks and points shared by the tests."""

import torch
import torch.nn as nn

NETS = {
    "logistic": lambda: nn.Sequential(nn.Linear(2, 1)),
    "bare_linear": lambda: nn.Linear(2, 1),
    "tanh3": lambda: nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)),
    "relu4": lambda: nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)),
    "sigmoid2": lambda: nn.Sequential(nn.Linear(2, 2), nn.Sigmoid(), nn.Linear(2, 1)),
    "two_hidden": lambda: nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 1)),
    "no_activation": lambda: nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1)),
    "identity": lambda: nn.Sequential(nn.Linear(2, 2), nn.Identity(), nn.ReLU(), nn.Linear(2, 1)),
    "three_inputs": lambda: nn.Sequential(nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 1)),
    "one_input": lambda: nn.Sequential(nn.Linear(1, 2), nn.Tanh(), nn.Linear(2, 1)),
    "wide": lambda: nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 1)),  # 49 weights: the drop-down panel
}
MULTICLASS = {  # networks with one output per class
    "three_classes": lambda: nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 3)),
    "softmax_only": lambda: nn.Sequential(nn.Linear(2, 4)),
    "two_hidden_classes": lambda: nn.Sequential(
        nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 3)
    ),
    "two_outputs": lambda: nn.Sequential(nn.Linear(2, 2), nn.Sigmoid(), nn.Linear(2, 2)),
}


def make(name, seed=0):
    torch.manual_seed(seed)
    return {**NETS, **MULTICLASS}[name]()


def class_points(net, n=9, seed=3):
    """n random points with the right number of inputs, and random labels 0 to K-1 for a network with K outputs."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, linears(net)[0].in_features, generator=g)
    k = linears(net)[-1].out_features
    return X, torch.randint(0, k, (n,), generator=g).float()


def linears(net):
    return [m for m in (net if isinstance(net, nn.Sequential) else [net]) if isinstance(m, nn.Linear)]


def points(net, n=7, seed=1):
    """n random points with the right number of inputs, and random classes."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, linears(net)[0].in_features, generator=g)
    y = (torch.rand(n, generator=g) < 0.5).float()
    return X, y


def naming_order(net):
    """The order of the names, from named_parameters alone: weight[j, :] then bias[j], unit by unit, layer by layer."""
    params = dict(net.named_parameters())
    out = []
    for name in params:
        if name.endswith("weight"):
            W, b = params[name], params[name[: -len("weight")] + "bias"]
            for j in range(W.shape[0]):
                out += [W[j, i].item() for i in range(W.shape[1])] + [b[j].item()]
    return out


def regression_points(net, n=9, seed=2):
    """n random points and real-valued targets."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, linears(net)[0].in_features, generator=g)
    return X, torch.randn(n, generator=g)


def case_slope_by_offset(model, X, y, k):
    """Oracle for the per-case slopes along weight k: a per-case offset t_i added to weight k inside the forward pass;
    one backward pass to t gives every case's slope of its own loss."""
    import torch.nn.functional as F

    lk, j, i = model.where[k]
    n = X.shape[0]
    t = torch.zeros(n, dtype=model.dtype, requires_grad=True)
    h = X
    for l, L in enumerate(model.layers):
        z = F.linear(h, L.linear.weight.detach(), L.linear.bias.detach())
        if l == lk:
            unit = torch.zeros(1, L.out_features, dtype=model.dtype)
            unit[0, j] = 1.0
            source = h[:, i : i + 1] if i >= 0 else torch.ones(n, 1, dtype=model.dtype)
            z = z + (t[:, None] * source) * unit
        h = L.module(z) if L.module is not None else z
    losses = model.loss(model.scores(h), y, reduction="none")
    (g,) = torch.autograd.grad(losses.sum(), t)
    return g.numpy()


def place_labels_reference(edges, sizes, obstacles, defaults, pad=0.0):
    """The original pure-Python label placement, kept as the oracle for the vectorised one."""
    from slopewalk.layout import LABEL_FRACTIONS, LABEL_SHIFTS, overlap_area

    placed, positions = [], []
    for (p0, p1), (lw, lh), t0 in zip(edges, sizes, defaults):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        nx, ny = -dy / length, dx / length
        best = None
        for t in LABEL_FRACTIONS:
            for shift in LABEL_SHIFTS:
                shift = shift * lh
                cx, cy = p0[0] + t * dx + shift * nx, p0[1] + t * dy + shift * ny
                box = (cx - lw / 2 - pad, cy - lh / 2 - pad, cx + lw / 2 + pad, cy + lh / 2 + pad)
                score = sum(overlap_area(box, o) for o in obstacles) + sum(overlap_area(box, q) for q in placed)
                score += 1e-4 * (abs(t - t0) + abs(shift))
                if best is None or score < best[0]:
                    best = (score, (cx, cy), box)
        placed.append(best[2])
        positions.append(best[1])
    return positions
