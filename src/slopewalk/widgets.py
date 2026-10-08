"""The controls of explore: sliders for the weights that move, eta, the step buttons and reset, a strip of readouts,
a "More views" section, and the picture. Needs ipywidgets."""

import math
import os

from .picture import LIVE_DPI, Picture
from .views import VIEWS

VIEW_LABELS = {
    "cases": "each case's slope",
    "slice": "R over two weights",
    "line": "R along the next step",
    "hidden": "the hidden layer",
    "momentum": "momentum",
}


def slider_range(v, lo=-4.0, hi=4.0):
    """The range of a weight's slider: lo to hi, widened on the side that v leaves, to floor(v) - 1 or ceil(v) + 1."""
    if v < lo:
        lo = math.floor(v) - 1.0
    if v > hi:
        hi = math.ceil(v) + 1.0
    return lo, hi


def name_ranges(names):
    """w1, w2, w3, w5 -> "w1 to w3, w5"."""
    nums = sorted(int(n[1:]) for n in names)
    parts, start = [], None
    for i, k in enumerate(nums):
        if start is None:
            start = k
        if i + 1 == len(nums) or nums[i + 1] != k + 1:
            parts.append(f"w{start}" if start == k else f"w{start} to w{k}")
            start = None
    return ", ".join(parts)


def readout_html(r):
    """The readout strip: one small card per number; an increase of R in the last step is shown in red."""

    def card(label, value, colour="#222"):
        return (
            '<div style="border:1px solid #DDD;border-radius:6px;padding:3px 10px;margin:2px 6px 2px 0;'
            f'background:#FAFAFA"><div style="font-size:11px;color:#666">{label}</div>'
            f'<div style="font-size:15px;font-family:monospace;color:{colour}">{value}</div></div>'
        )

    cards = [card("R", f"{r['R']:.4f}"), card("size of the gradient", f"{r['grad_norm']:.4f}")]
    if r["accuracy"] is not None:
        cards.append(card("accuracy", f"{r['accuracy']:.3f}"))
    cards.append(card(f"next step, eta {r['eta']:g}: predicted change of R", f"{r['next_change']:+.4f}"))
    if r["last"] is not None:
        predicted, actual = r["last"]
        colour = "#B00020" if actual > 0 else "#1B6E3A"
        cards.append(card("last step: predicted, actual change", f"{predicted:+.4f}, {actual:+.4f}", colour))
    return '<div style="display:flex;flex-wrap:wrap">' + "".join(cards) + "</div>"


class Controller:
    """Builds the widgets for a session and keeps them, the network and the picture in step."""

    def __init__(self, session, dpi=LIVE_DPI):
        import ipywidgets as widgets

        S = self.session = session
        self.dpi = dpi
        self.busy = False
        self.picture = None
        self.use_output = os.environ.get("SLOPEWALK_DISPLAY") == "output"  # fallback: the PNG through display()
        self.index = list(S.active)  # the weight of each slider
        self.sliders = []
        for k in self.index:
            lo, hi = slider_range(S.w0[k])
            self.sliders.append(
                widgets.FloatSlider(
                    value=S.w0[k],
                    min=lo,
                    max=hi,
                    step=0.001,
                    description=S.names[k],
                    readout_format=".2f",
                    continuous_update=False,
                    layout=widgets.Layout(width="230px"),
                )
            )
        panel = self.grouped_panel(widgets) if len(self.sliders) <= 30 else self.dropdown_panel(widgets)
        frozen = [n for k, n in enumerate(S.names) if k not in set(self.index)]
        if frozen:
            note = widgets.HTML(f"<i>fixed (drawn lighter, no slider): {name_ranges(frozen)}</i>")
            panel = widgets.VBox([panel, note])
        small = S.eta < 0.1
        self.eta = widgets.FloatSlider(
            value=S.eta,
            min=min(0.1, S.eta),
            max=max(5.0, S.eta),
            step=0.01 if small else 0.1,
            description="eta",
            readout_format=".2f" if small else ".1f",
        )
        self.buttons = {name: widgets.Button(description=name) for name in ("gradient step", "10 steps", "reset")}
        self.readout = widgets.HTML()
        self.image = widgets.Image(format="png", layout=widgets.Layout(max_width="100%", height="auto"))
        self.out = widgets.Output()
        self.more = self.views_panel(widgets)
        for s in self.sliders:
            s.observe(self.on_slider, names="value")
        self.eta.observe(self.on_eta, names="value")
        self.buttons["gradient step"].on_click(lambda _: self.steps(1))
        self.buttons["10 steps"].on_click(lambda _: self.steps(10))
        self.buttons["reset"].on_click(lambda _: self.reset())
        if S.layout.style == "wide":  # the controls in a column beside the picture
            column = widgets.VBox(
                [panel, self.eta, widgets.HBox(list(self.buttons.values())), self.readout, self.more]
                + ([] if self.use_output else [self.out]),
                layout=widgets.Layout(width="390px", min_width="390px"),
            )
            right = self.out if self.use_output else self.image
            right.layout.flex = "1 1 auto"
            self.widgets = [widgets.HBox([column, right], layout=widgets.Layout(align_items="flex-start"))]
        else:
            controls = widgets.HBox([self.eta, *self.buttons.values()])
            shown = [self.out] if self.use_output else [self.image, self.out]
            self.widgets = [panel, controls, self.more, self.readout, *shown]

    # ---------------------------------------------------------------- building

    def grouped_panel(self, widgets):
        """Up to 30 sliders: one box per unit with weights that move, headed by the names of their inputs."""
        S = self.session
        M = S.model
        moving = set(self.index)
        boxes, k = [], 0
        for layer, L in enumerate(M.layers):
            labels = M.input_labels(layer) + ["bias"]
            for j in range(L.out_features):
                ks = [k + i for i in range(L.in_features + 1) if k + i in moving]
                if ks:
                    names = ", ".join(labels[q - k] for q in ks)
                    head = widgets.HTML(f"<b>{M.unit_title(layer, j)}</b>: weights for {names}")
                    boxes.append(widgets.VBox([head] + [self.sliders[self.index.index(q)] for q in ks]))
                k += L.in_features + 1
        return widgets.Box(boxes, layout=widgets.Layout(flex_flow="row wrap"))

    def dropdown_panel(self, widgets):
        """More than 30 sliders: choose one weight, then move it."""
        names = [self.session.names[k] for k in self.index]
        pick = widgets.Dropdown(options=[(nm, i) for i, nm in enumerate(names)], value=0, description="weight")
        holder = widgets.HBox([self.sliders[0]])
        pick.observe(lambda ch: setattr(holder, "children", [self.sliders[ch["new"]]]), names="value")
        return widgets.HBox([pick, holder])

    def views_panel(self, widgets):
        """The "More views" section: a checkbox per view and their settings."""
        S = self.session
        names = list(S.names)
        self.view_boxes = {
            v: widgets.Checkbox(value=v in S.views, description=VIEW_LABELS[v], indent=False) for v in VIEWS
        }
        self.case_pick = widgets.Dropdown(options=names, value=names[S.case_weight], description="case slopes of")
        a = S.slice_weights[0]
        b = S.slice_weights[1] if len(S.slice_weights) > 1 else None
        self.slice_pick = [
            widgets.Dropdown(options=names, value=names[a], description="slice over"),
            widgets.Dropdown(options=["none"] + names, value="none" if b is None else names[b], description="and"),
        ]
        self.beta = widgets.FloatSlider(value=S.beta, min=0.0, max=0.99, step=0.01, description="beta")
        self.ahead = widgets.IntSlider(value=S.lookahead_steps, min=1, max=200, description="steps ahead")
        self.log_box = widgets.Checkbox(value=S.log_scale, description="R on a log scale", indent=False)
        for w in [*self.view_boxes.values(), self.case_pick, *self.slice_pick, self.beta, self.ahead, self.log_box]:
            w.observe(self.on_views, names="value")
        box = widgets.VBox(
            [
                widgets.HBox(
                    list(self.view_boxes.values()) + [self.log_box], layout=widgets.Layout(flex_flow="row wrap")
                ),
                widgets.HBox([self.case_pick, *self.slice_pick]),
                widgets.HBox([self.beta, self.ahead]),
            ]
        )
        acc = widgets.Accordion(children=[box])
        acc.set_title(0, "More views")
        acc.selected_index = None
        return acc

    # ---------------------------------------------------------------- state

    def values(self):
        """The full weight vector: the sliders for the weights that move, the network for the others."""
        w = self.session.weights()
        for k, s in zip(self.index, self.sliders):
            w[k] = s.value
        return w

    def on_slider(self, *_):
        if self.busy:
            return
        self.session.model.set_weights(self.values())
        self.redraw()

    def on_eta(self, change):
        self.session.eta = float(change["new"])
        if not self.busy:
            self.redraw()

    def on_views(self, *_):
        if self.busy:
            return
        S = self.session
        names = list(S.names)
        S.views = tuple(v for v in VIEWS if self.view_boxes[v].value)
        S.case_weight = names.index(self.case_pick.value)
        first = names.index(self.slice_pick[0].value)
        second = self.slice_pick[1].value
        S.slice_weights = [first] if second in ("none", self.slice_pick[0].value) else [first, names.index(second)]
        S.beta = float(self.beta.value)
        S.lookahead_steps = int(self.ahead.value)
        S.log_scale = bool(self.log_box.value)
        self.picture = None  # the panels changed: a new figure
        self.redraw()

    def sync(self):
        """After steps, reset or record in code: the sliders show the network's weights, then one redraw."""
        self.busy = True
        try:
            w = self.session.weights()
            for k, s in zip(self.index, self.sliders):
                lo, hi = slider_range(w[k], s.min, s.max)
                if hi > s.max:
                    s.max = hi
                if lo < s.min:
                    s.min = lo
                s.value = w[k]
        finally:
            self.busy = False
        self.redraw()

    def steps(self, m):
        """m gradient steps from the slider values with the current eta; the session then updates the sliders."""
        S = self.session
        S.eta = float(self.eta.value)
        S.step(m, S.eta, self.values())

    def reset(self):
        """Back to the weights at the start, with no steps taken, and the sliders' starting ranges."""
        S = self.session
        S.reset()
        self.busy = True
        try:
            for k, s in zip(self.index, self.sliders):
                lo, hi = slider_range(S.w0[k])
                s.min, s.max = lo, hi
        finally:
            self.busy = False

    def redraw(self):
        S = self.session
        with self.out:
            if self.picture is None:
                self.picture = Picture(S, dpi=self.dpi)
            w = self.values()
            self.picture.refresh(w)
            S.fig = self.picture.fig
            png = self.picture.png()
            self.readout.value = readout_html(S.readouts(self.eta.value, w))
            if self.use_output:
                from IPython.display import Image, clear_output, display

                clear_output(wait=True)
                display(Image(png))
            else:
                self.image.value = png

    def show(self):
        from IPython.display import display

        display(*self.widgets)
        self.redraw()
        return self.session
