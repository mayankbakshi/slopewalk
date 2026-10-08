"""Points for explore: clusters around centres, the built-in examples, and the checks on the points of a call."""

import numpy as np
import torch

XOR_CENTRES = [(-1, 1), (1, 1), (-1, -1), (1, -1)]  # A, B, C, D: top left, top right, bottom left, bottom right
XOR_LABELS = [1, 0, 0, 1]
REGRESSION_GROUPS = 4  # regression with more than 8 points: this many groups of neighbouring points along x1


def clusters(centers, labels, noise=0.0, n_per=50, seed=0):
    """Points for explore: the centres themselves (noise = 0), or n_per points around each centre, with standard
    deviation noise. Returns X (n x d), y (n) and groups (n): the number of the centre of each point."""
    C = torch.tensor(centers, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.float32)
    if noise > 0:
        g = torch.Generator().manual_seed(seed)
        X = torch.cat([c + noise * torch.randn(n_per, C.shape[1], generator=g) for c in C])
        return X, Y.repeat_interleave(n_per), torch.arange(len(C)).repeat_interleave(n_per)
    return C, Y, torch.arange(len(C))


def xor_centres(noise=0.0, n_per=50, seed=0):
    """The four XOR centres A to D (class 1 where x1 and x2 differ in sign), or clusters around them."""
    return clusters(XOR_CENTRES, XOR_LABELS, noise, n_per, seed)


def two_points():
    """A = (-1, 1) of class 1 and B = (1, 1) of class 0: the warm-up without a hidden layer."""
    return clusters([(-1, 1), (1, 1)], [1, 0])


def sine_points(n=16, noise=0.1, seed=0):
    """A regression example with one input: n points on y = sin(1.5 x) for x from -2 to 2, plus Gaussian noise with
    standard deviation noise. Returns X (n x 1), y (n) and groups (n): four groups of neighbouring points along x.
    Use it with task="regression"."""
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(-2, 2, n)
    y = torch.sin(1.5 * x) + noise * torch.randn(n, generator=g)
    return x[:, None], y, torch.arange(n) * REGRESSION_GROUPS // n


class Data:
    """The points of one explore call: X (n x d), y (n), groups (n), and which points belong to each group."""

    def __init__(self, X, y, groups, task="binary"):
        self.X, self.y, self.groups, self.task = X, y, groups, task
        self.n, self.d = X.shape
        self.gids = sorted(set(groups.tolist()))
        self.G = len(self.gids)
        self.gsel = [(groups == g).numpy() for g in self.gids]

    def stats(self, arr, c):
        """Mean and standard deviation over the points of group c of the rows of arr; the sd is 0 for a single point."""
        sel = arr[self.gsel[c]]
        return sel.mean(0), (sel.std(0) if sel.shape[0] > 1 else 0 * sel.mean(0))


def as_numbers(v, name):
    """v as a float64 array, or ValueError naming the argument."""
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    try:
        a = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"slopewalk: {name} must hold numbers: a list of points of equal length, an array or a tensor."
        ) from e
    return a


def check_finite(a, name):
    bad = ~np.isfinite(a)
    if bad.any():
        rows = np.unique(np.nonzero(bad)[0])
        raise ValueError(
            f"slopewalk: {name} contains NaN or inf at {len(rows)} point(s), the first at index {rows[0]}. "
            "Remove those points or replace the values."
        )


def infer_task(y, task):
    """The task: as given, or binary when every label is 0 or 1. Raises ValueError rather than guess."""
    binary = bool(np.isin(y, (0.0, 1.0)).all())
    if task not in (None, "binary", "regression"):
        raise ValueError(f"slopewalk: task must be 'binary' or 'regression', not {task!r}.")
    if task == "regression":
        return task
    if not binary:
        other = y[~np.isin(y, (0.0, 1.0))][0]
        raise ValueError(
            f"slopewalk: y has values other than 0 and 1 (for example {other:g}). For classification the labels must "
            "be 0 or 1; for a numerical target pass task='regression'."
        )
    return "binary"


def prepare(X, y, groups=None, n_inputs=None, dtype=torch.float32, task=None):
    """The checked tensors of an explore call, as a Data object.

    X: n points with n_inputs numbers each (a list of points, an array or a tensor). A flat X is one point, except for
    a network with one input, where it is n points. y: n labels (0 or 1) or targets (task='regression').
    groups None: one group per point up to 8 points; more points: one group per class (binary), or for regression
    four groups of neighbouring points along x1 (one input) or a single group (several inputs).
    """
    Xa = as_numbers(X, "X")
    if Xa.ndim == 0:
        raise ValueError("slopewalk: X must hold at least one point.")
    if Xa.ndim == 1:
        if n_inputs == 1:
            Xa = Xa[:, None]
        elif n_inputs is None or Xa.shape[0] == n_inputs:
            Xa = Xa[None, :]
        else:
            raise ValueError(
                f"slopewalk: X has {Xa.shape[0]} numbers, but the network has {n_inputs} inputs. Give one point as "
                f"{n_inputs} numbers or several points as an n x {n_inputs} array."
            )
    if Xa.ndim != 2:
        raise ValueError(f"slopewalk: X must be an n x d array of points; it has shape {Xa.shape}.")
    n, d = Xa.shape
    if n == 0:
        raise ValueError("slopewalk: X has no points.")
    if n_inputs is not None and d != n_inputs:
        raise ValueError(f"slopewalk: the points of X have {d} inputs, but the network has {n_inputs}.")
    ya = as_numbers(y, "y").reshape(-1)
    if ya.shape[0] != n:
        raise ValueError(f"slopewalk: y has {ya.shape[0]} values for {n} points; give one label per point.")
    check_finite(Xa, "X")
    check_finite(ya, "y")
    task = infer_task(ya, task)
    if groups is None:
        if n <= 8:
            ga = np.arange(n)
        elif task == "binary":
            ga = ya.astype(np.int64)
        elif d == 1:
            order = np.argsort(Xa[:, 0], kind="stable")
            ga = np.empty(n, dtype=np.int64)
            ga[order] = np.arange(n) * REGRESSION_GROUPS // n
        else:
            ga = np.zeros(n, dtype=np.int64)
    else:
        g = as_numbers(groups, "groups").reshape(-1)
        if g.shape[0] != n:
            raise ValueError(f"slopewalk: groups has {g.shape[0]} values for {n} points; give one number per point.")
        if not (np.isfinite(g).all() and (g == np.round(g)).all()):
            raise ValueError("slopewalk: groups must hold one whole number per point.")
        ga = g.astype(np.int64)
    Xt = torch.as_tensor(np.ascontiguousarray(Xa), dtype=dtype)  # reversed or sliced arrays need a copy
    yt = torch.as_tensor(np.ascontiguousarray(ya), dtype=dtype)
    return Data(Xt, yt, torch.as_tensor(np.ascontiguousarray(ga), dtype=torch.int64), task)
