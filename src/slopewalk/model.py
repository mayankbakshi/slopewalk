"""The network: parsing nn.Sequential, the weights as one list, forward values, the loss and the slopes."""

import re
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# activation module -> (name, numpy function, half-width of the curve drawn in the glyph, numpy derivative)
ACTIVATIONS = {
    nn.Tanh: ("tanh", np.tanh, 3.0, lambda t: 1 - np.tanh(t) ** 2),
    nn.ReLU: ("ReLU", lambda t: np.maximum(t, 0), 3.0, lambda t: (np.asarray(t) > 0).astype(float)),
    nn.Sigmoid: (
        "sigmoid",
        lambda t: 1 / (1 + np.exp(-t)),
        6.0,
        lambda t: np.exp(-t) / (1 + np.exp(-t)) ** 2,
    ),
}
SIGMOID = ACTIVATIONS[nn.Sigmoid]
SUPPORTED = "nn.Linear with nn.Tanh, nn.ReLU or nn.Sigmoid between them"
TASKS = ("binary", "regression")
BATCH_CELLS = 400_000  # risk_batch handles at most this many (weight vector, point) pairs at once

Weight = namedtuple("Weight", "name layer unit input value")
# hidden: [(z, h), ...] per hidden layer; a: the score (binary) or the prediction (regression); p: sigmoid(a) for
# binary, None for regression; loss: per point
Values = namedtuple("Values", "hidden a p loss")


class Layer:
    """An nn.Linear and the activation after it: act is (name, f, lim) or None, module the activation module or None."""

    def __init__(self, linear):
        self.linear = linear
        self.act = None
        self.module = None

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features


def parse_layers(net):
    """The layers of net, or ValueError when net is not of the supported kind."""
    mods = list(net) if isinstance(net, nn.Sequential) else [net]
    layers = []
    for m in mods:
        if isinstance(m, nn.Linear):
            layers.append(Layer(m))
        elif type(m) in ACTIVATIONS and layers and layers[-1].act is None:
            layers[-1].act = ACTIVATIONS[type(m)]
            layers[-1].module = m
        elif isinstance(m, nn.Identity):
            continue
        else:
            raise ValueError(f"slopewalk: unsupported layer {m}. Use {SUPPORTED}.")
    if not layers or layers[-1].out_features != 1 or layers[-1].act is not None:
        raise ValueError(
            "slopewalk: the network must end with nn.Linear(..., 1) (the score of class 1, or the prediction for "
            "task='regression'), without an activation."
        )
    for L in layers:
        if L.linear.bias is None:
            raise ValueError(f"slopewalk: {L.linear} has no bias. Use nn.Linear with bias=True.")
    return layers


class Model:
    """The parsed network.

    The weights are one list in the order of their names: layer by layer, unit by unit, each unit's incoming weights
    in the order of its inputs, then its bias. For nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)) this
    gives w1 to w3 (unit 1), w4 to w6, w7 to w9, w10 to w12 (output weights) and w13 (output bias).
    task: "binary" (R is the log loss, as F.binary_cross_entropy_with_logits) or "regression" (R is the mean
    squared error, as F.mse_loss).
    """

    def __init__(self, net, task="binary"):
        if task not in TASKS:
            raise ValueError(f"slopewalk: task must be 'binary' or 'regression', not {task!r}.")
        self.net = net
        self.task = task
        self.layers = parse_layers(net)
        self.linears = [L.linear for L in self.layers]
        tensors = [t for L in self.linears for t in (L.weight, L.bias)]
        if any(t.device.type != "cpu" for t in tensors):
            raise ValueError("slopewalk works on networks on the CPU. Move the network there first: net.cpu().")
        dtypes = {t.dtype for t in tensors}
        if len(dtypes) > 1 or not dtypes <= {torch.float32, torch.float64}:
            found = ", ".join(sorted(str(d).replace("torch.", "") for d in dtypes))
            raise ValueError(
                f"slopewalk: the network's weights are {found}. Use float32 (the default) or float64 for all layers: "
                "net.float() or net.double()."
            )
        self.dtype = tensors[0].dtype
        self.n_inputs = self.layers[0].in_features
        self.hidden = [L.out_features for L in self.layers[:-1]]
        self.n_weights = sum((L.in_features + 1) * L.out_features for L in self.layers)
        self.names = [f"w{k + 1}" for k in range(self.n_weights)]
        # where each weight sits: (layer, unit, input), input -1 for the bias
        self.where = [
            (l, j, i)
            for l, L in enumerate(self.layers)
            for j in range(L.out_features)
            for i in list(range(L.in_features)) + [-1]
        ]

    # ---------------------------------------------------------------- names and places

    def unit_name(self, layer, j):
        """The index written after z and h: "2" with one hidden layer, "1,2" with several."""
        return f"{j + 1}" if len(self.hidden) == 1 else f"{layer + 1},{j + 1}"

    def unit_title(self, layer, j):
        if layer == len(self.layers) - 1:
            return "output unit"
        return f"unit {j + 1}" if len(self.hidden) == 1 else f"layer {layer + 1}, unit {j + 1}"

    def input_labels(self, layer):
        """What feeds the units of a layer: x1, x2, ... for the first, h1, h2, ... (or h1,1, h1,2, ...) after it."""
        n = self.layers[layer].in_features
        if layer == 0:
            return [f"x{i + 1}" for i in range(n)]
        return [f"h{i + 1}" if len(self.hidden) == 1 else f"h{layer},{i + 1}" for i in range(n)]

    def layer_indices(self, layer):
        """The indices of the weights of one layer (from 0)."""
        return [k for k, (l, _, _) in enumerate(self.where) if l == layer]

    def select(self, params):
        """The indices of the weights that move: params None (all), a list of names, a string of names separated by
        commas, "output" (the weights of the output unit) or "layer:k" (the weights of layer k, from 1)."""
        if params is None:
            return list(range(self.n_weights))
        n_layers = len(self.layers)
        if isinstance(params, str):
            text = params.strip()
            if text == "output":
                return self.layer_indices(n_layers - 1)
            m = re.fullmatch(r"layer\s*:\s*(\d+)", text)
            if m:
                k = int(m.group(1))
                if not 1 <= k <= n_layers:
                    raise ValueError(f"slopewalk: params='{text}': this network has layers 1 to {n_layers}.")
                return self.layer_indices(k - 1)
            params = [p.strip() for p in text.split(",") if p.strip()]
        names = list(params)
        unknown = [p for p in names if p not in self.names]
        if unknown or not names:
            raise ValueError(
                f"slopewalk: params {unknown or names} does not name weights of this network. Use names from w1 to "
                f"w{self.n_weights}, 'output', or 'layer:k' with k from 1 to {n_layers}."
            )
        return sorted({self.names.index(p) for p in names})

    # ---------------------------------------------------------------- weights in and out

    def get_weights(self):
        """The weights as Python floats, in the order of their names."""
        return [
            v
            for L in self.linears
            for j in range(L.out_features)
            for v in L.weight[j].detach().tolist() + [L.bias[j].item()]
        ]

    def set_weights(self, w):
        """Writes w (in the order of the names) into the network, in place."""
        k = 0
        with torch.no_grad():
            for L in self.linears:
                m = L.in_features
                for j in range(L.out_features):
                    L.weight[j].copy_(torch.tensor(w[k : k + m], dtype=L.weight.dtype))
                    L.bias[j].fill_(float(w[k + m]))
                    k += m + 1

    def weight_table(self):
        rows, k = [], 0
        w = self.get_weights()
        for layer, L in enumerate(self.layers):
            inputs = self.input_labels(layer) + ["bias"]
            for j in range(L.out_features):
                for name in inputs:
                    rows.append(Weight(self.names[k], layer + 1, j + 1, name, w[k]))
                    k += 1
        return rows

    # ---------------------------------------------------------------- loss, values, R and slopes

    def loss(self, a, y, reduction="mean"):
        """The loss of the task: log loss on the score a (binary) or squared error of the prediction a (regression)."""
        if self.task == "regression":
            return F.mse_loss(a, y, reduction=reduction)
        return F.binary_cross_entropy_with_logits(a, y, reduction=reduction)

    def forward(self, X, y, w=None):
        """Values at w (default: the current weights): (z, h) per hidden layer, then a, p and the loss, per point."""
        if w is not None:
            self.set_weights(w)
        hidden = []
        with torch.no_grad():
            t = X
            for L in self.layers[:-1]:
                z = L.linear(t)
                t = L.module(z) if L.module is not None else z
                hidden.append((z.numpy(), t.numpy()))
            a = self.layers[-1].linear(t).squeeze(1)
            p = torch.sigmoid(a).numpy() if self.task == "binary" else None
            loss = self.loss(a, y, reduction="none")
        return Values(hidden, a.numpy(), p, loss.numpy())

    def risk(self, X, y, w=None):
        """R: the average loss over all points (log loss for binary, squared error for regression)."""
        if w is not None:
            self.set_weights(w)
        with torch.no_grad():
            return float(self.loss(self.net(X).reshape(-1), y))

    def slopes(self, X, y, w=None):
        """dR/dw for every weight, from R.backward(), in the order of the names.

        Leaves the network as it was: the .grad of every parameter and its requires_grad flag are untouched.
        """
        if w is not None:
            self.set_weights(w)
        params = [p for L in self.linears for p in (L.weight, L.bias)]
        flags = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(True)
        try:
            with torch.enable_grad():
                R = self.loss(self.net(X).reshape(-1), y)
                grads = torch.autograd.grad(R, params)
        finally:
            for p, flag in zip(params, flags):
                p.requires_grad_(flag)
        return np.array(
            [
                v
                for gw, gb in zip(grads[0::2], grads[1::2])
                for j in range(gw.shape[0])
                for v in gw[j].tolist() + [gb[j].item()]
            ]
        )

    def risk_batch(self, X, y, W):
        """R for every row of W (G x n_weights), in one batched forward pass; the network is not changed."""
        W = torch.as_tensor(np.asarray(W, dtype=np.float64), dtype=self.dtype)
        if W.dim() == 1:
            W = W[None, :]
        n = X.shape[0]
        chunk = max(1, BATCH_CELLS // max(n, 1))
        out = []
        with torch.no_grad():
            for start in range(0, W.shape[0], chunk):
                Wc = W[start : start + chunk]
                G = Wc.shape[0]
                t = X.expand(G, *X.shape)
                k = 0
                for L in self.layers:
                    m, o = L.in_features, L.out_features
                    block = Wc[:, k : k + o * (m + 1)].reshape(G, o, m + 1)
                    z = torch.einsum("gni,goi->gno", t, block[:, :, :m]) + block[:, None, :, m]
                    t = L.module(z) if L.module is not None else z
                    k += o * (m + 1)
                losses = self.loss(t.reshape(G, n), y.expand(G, n), reduction="none")
                out.append(losses.mean(1))
        return torch.cat(out).numpy().astype(float)

    def case_slopes(self, X, y, w=None):
        """For every case i and weight k, the slope of case i's own loss along w_k, as an n x n_weights array.

        The mean over the cases is dR/dw. One backward pass: the summed losses are differentiated with respect to
        every layer's pre-activations z; since z_i only affects loss_i, this gives each case's slope with respect
        to each unit, and the product with the unit's inputs (and 1 for the bias) gives the slope along each weight.
        The network's parameters, their .grad and their requires_grad flags are not touched.
        """
        if w is not None:
            self.set_weights(w)
        with torch.enable_grad():
            h = X.detach()
            inputs, zs = [], []
            for L in self.layers:
                inputs.append(h)
                z = F.linear(h, L.linear.weight.detach(), L.linear.bias.detach())
                if not z.requires_grad:
                    z.requires_grad_(True)
                zs.append(z)
                h = L.module(z) if L.module is not None else z
            losses = self.loss(h.reshape(-1), y, reduction="none")
            deltas = torch.autograd.grad(losses.sum(), zs)
        n = X.shape[0]
        blocks = []
        for h_in, d in zip(inputs, deltas):
            with_one = torch.cat([h_in.detach(), torch.ones(n, 1, dtype=h_in.dtype)], dim=1)  # inputs, then 1
            blocks.append((d[:, :, None] * with_one[:, None, :]).reshape(n, -1))  # unit by unit
        return torch.cat(blocks, dim=1).detach().numpy().astype(float)

    def accuracy(self, X, y, w=None):
        """The share of cases on the correct side of the boundary (binary only)."""
        if w is not None:
            self.set_weights(w)
        with torch.no_grad():
            a = self.net(X).reshape(-1)
        return float(((a >= 0).to(y.dtype) == y).to(torch.float64).mean())


def weight_table(net):
    """One row per weight: name, layer (from 1), unit (from 1), input ("x1", "h2", "h1,2" or "bias") and value."""
    return Model(net).weight_table()
