"""The rule of a gradient step: plain gradient descent, heavy-ball momentum or Adam, with an optional weight decay.

Every rule moves only the weights that move (params) and leaves R as it is: R stays the mean loss on the data, so
the slopes in the table and the backward pass are the slopes of R; the weight decay is a term lambda * w added to each
slope before the step, as torch.optim does with weight_decay. The step of each rule, with g' = dR/dw + lambda * w:

    gd        dw = -eta * g'
    momentum  v <- beta * v - eta * g';  dw = v                   (each step keeps the share beta of the last one)
    adam      m <- beta * m + (1 - beta) * g';  v <- beta2 * v + (1 - beta2) * g'^2;  t <- t + 1
              dw = -eta * (m / (1 - beta^t)) / (sqrt(v / (1 - beta2^t)) + eps)

The first-order prediction of a step is g . dw (for plain gradient descent, -eta * |g|^2).
"""

import math

OPTIMIZERS = ("gd", "momentum", "adam")


def check_optimizer(kind):
    if kind not in OPTIMIZERS:
        raise ValueError(f"slopewalk: optimizer must be 'gd', 'momentum' or 'adam', not {kind!r}.")
    return kind


def check_decay(weight_decay):
    if not (isinstance(weight_decay, (int, float)) and math.isfinite(weight_decay) and weight_decay >= 0):
        raise ValueError(f"slopewalk: weight_decay must be a number from 0 up, not {weight_decay!r}.")
    return float(weight_decay)


def check_beta2(beta2):
    if not (isinstance(beta2, (int, float)) and 0 <= beta2 < 1):
        raise ValueError(
            f"slopewalk: beta2 (Adam's share of the last squared slopes) must be in [0, 1), not {beta2!r}."
        )
    return float(beta2)


class Optimizer:
    """The step rule with its state: the velocity of momentum, or Adam's m, v and step count t."""

    def __init__(self, kind="gd", weight_decay=0.0, beta=0.9, beta2=0.999, eps=1e-8):
        self.kind = check_optimizer(kind)
        self.weight_decay = check_decay(weight_decay)
        self.beta, self.beta2, self.eps = beta, check_beta2(beta2), eps
        self.reset()

    def reset(self):
        """Forgets the steps taken: the next step is a first step."""
        self.velocity = {}  # momentum: per weight
        self.m, self.v, self.t = {}, {}, 0  # Adam

    def copy(self):
        """A copy of the state, for back one step."""
        out = Optimizer(self.kind, self.weight_decay, self.beta, self.beta2, self.eps)
        out.velocity, out.m, out.v, out.t = dict(self.velocity), dict(self.m), dict(self.v), self.t
        return out

    def set(self, kind=None, weight_decay=None, beta=None, beta2=None):
        """Changes the rule or its numbers; a new rule forgets the state."""
        if kind is not None and kind != self.kind:
            self.kind = check_optimizer(kind)
            self.reset()
        if weight_decay is not None:
            self.weight_decay = check_decay(weight_decay)
        if beta is not None:
            self.beta = float(beta)
        if beta2 is not None:
            self.beta2 = check_beta2(beta2)

    def direction(self, g, w, active, eta):
        """The change dw of each weight that the next step would make, as {k: dw}, without taking it."""
        return self.take(g, w, active, eta, self.copy())

    def step(self, g, w, active, eta):
        """The change dw of each weight for this step, as {k: dw}; the state moves on."""
        return self.take(g, w, active, eta, self)

    @staticmethod
    def take(g, w, active, eta, state):
        """The step of the rule in `state` (whose state moves on), for the weights in `active`."""
        out = {}
        lam, kind = state.weight_decay, state.kind
        if kind == "adam":
            state.t += 1
            b1, b2 = state.beta, state.beta2
            fix1, fix2 = 1 - b1**state.t, 1 - b2**state.t
        for k in active:
            gk = float(g[k]) + lam * float(w[k])
            if kind == "gd":
                out[k] = -eta * gk
            elif kind == "momentum":
                state.velocity[k] = state.beta * state.velocity.get(k, 0.0) - eta * gk
                out[k] = state.velocity[k]
            else:
                state.m[k] = b1 * state.m.get(k, 0.0) + (1 - b1) * gk
                state.v[k] = b2 * state.v.get(k, 0.0) + (1 - b2) * gk * gk
                out[k] = -eta * (state.m[k] / fix1) / (math.sqrt(state.v[k] / fix2) + state.eps)
        return out
