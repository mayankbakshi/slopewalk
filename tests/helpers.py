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


def make(name, seed=0):
    torch.manual_seed(seed)
    return NETS[name]()


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
