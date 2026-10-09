"""The controls of explore. The picture comes first; under it a status line (the key numbers and one sentence on the
last action), then the controls: the sliders grouped by unit, and one row with eta, the step buttons and the layout.
Tabs appear only for what the explore call asks for: "Backprop" when the views include "backprop", "Views" when
other views are asked for. Changing tabs leaves the picture as it is. Needs ipywidgets.
"""

import math
import os

from .picture import LIVE_DPI, Picture
from .views import VIEWS

TITLES = {"core": "Weights and steps", "backprop": "Backprop", "views": "Views"}
VIEW_LABELS = {
    "backprop": "backward pass",
    "cases": "each case's slope",
    "slice": "R over two weights",
    "line": "R along the next step",
    "hidden": "the hidden layer",
    "momentum": "momentum",
}
VIEW_HELP = {
    "cases": "every case's slope along one weight; their average is the slope in the table",
    "slice": "contours of R over two weights, with the path of the steps",
    "line": "R along the next step, next to its first-order prediction",
    "hidden": "the cases at their hidden values, with the output unit's boundary",
    "momentum": "the next steps of gradient descent and of momentum",
}
LOOK = {  # where to look in the picture at each stage of the backward pass
    "forward": "The picture shows the forward pass. Press next stage to start at the output.",
    "output": "Under a, the purple box is δa = p - y of each case.",
    "weights": "On the purple lines, each number is δ of the unit times the input of its line.",
    "to_h": "The purple arrows carry δ back along the weights to each h.",
    "delta": "The purple tangents on the curves: their slope multiplies ∂ℓ/∂h to give δ.",
    "average": "The average of these slopes over the cases is the slope column of the table, what R.backward() gives.",
}
HINTS = {
    "core": "Press gradient step to move every weight against its slope, by eta times the slope. Drag a slider to "
    "change one weight by hand: each column holds the weights into one unit.",
    "backprop": "Step through the backward pass, from the output back to the first layer. The panel under the picture "
    "writes out the numbers for the case chosen in trace case.",
    "views": "Tick a box to add a panel under the network.",
}
HINT_STYLE = "font-size:13px;color:#555;margin:2px 0 6px 0"
SLIDER_STYLE = {"description_width": "40px"}


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
    """The numbers of the status line, as a plain table: a row of labels over a row of values."""
    cells = [("R", f"{r['R']:.4f}", "#222"), ("size of the gradient", f"{r['grad_norm']:.4f}", "#222")]
    if r["accuracy"] is not None:
        cells.append(("accuracy", f"{r['accuracy']:.3f}", "#222"))
    cells.append((f"next step, eta {r['eta']:g}: predicted change of R", f"{r['next_change']:+.4f}", "#222"))
    if r["last"] is not None:
        predicted, actual = r["last"]
        colour = "#B00020" if actual > 0 else "#1B6E3A"
        cells.append(("last step: predicted, actual change", f"{predicted:+.4f}, {actual:+.4f}", colour))
    pad = "padding:1px 28px 1px 0;text-align:left"
    head = "".join(f'<td style="{pad};color:#666">{label}</td>' for label, _, _ in cells)
    vals = "".join(f'<td style="{pad};font-family:monospace;color:{c}">{v}</td>' for _, v, c in cells)
    return f'<table style="border-collapse:collapse;font-size:13px"><tr>{head}</tr><tr>{vals}</tr></table>'


def message_html(text):
    return f'<div style="font-size:14px;color:#222;margin:6px 0 2px 0;min-height:1.4em">{text}</div>'


def text_html(text):
    return f'<div style="{HINT_STYLE}">{text}</div>'


def tip(widget, text):
    """A tooltip, on ipywidgets 7 and 8."""
    if hasattr(widget, "tooltip"):
        widget.tooltip = text
    elif hasattr(widget, "description_tooltip"):
        widget.description_tooltip = text
    return widget


def change_words(before, after):
    d = after - before
    if abs(d) < 5e-5:
        return f"R stayed at {after:.4f}"
    return f"R went from {before:.4f} to {after:.4f}, {'down' if d < 0 else 'up'} {abs(d):.4f}"


class Controller:
    """Builds the widgets for a session and keeps them, the network and the picture in step."""

    def __init__(self, session, dpi=LIVE_DPI):
        import ipywidgets as widgets

        S = self.session = session
        self.dpi = dpi
        self.busy = False
        self.picture = None
        self.last_R = None
        self.use_output = os.environ.get("SLOPEWALK_DISPLAY") == "output"  # fallback: the PNG through display()
        self.index = list(S.active)  # the weight of each slider
        self.sliders = []
        for k in self.index:
            lo, hi = slider_range(S.w0[k])
            slider = widgets.FloatSlider(
                value=S.w0[k],
                min=lo,
                max=hi,
                step=0.001,
                description=S.names[k],
                readout_format=".2f",
                continuous_update=False,
                style=SLIDER_STYLE,
                layout=widgets.Layout(width="250px"),
            )
            self.sliders.append(tip(slider, f"{S.names[k]}: drag to change this weight in the network"))
        self.panel = self.grouped_panel(widgets) if len(self.sliders) <= 30 else self.dropdown_panel(widgets)
        frozen = [n for k, n in enumerate(S.names) if k not in set(self.index)]
        self.frozen_note = widgets.HTML(text_html(f"Fixed, drawn lighter: {name_ranges(frozen)}.") if frozen else "")

        small = S.eta < 0.1
        self.eta = widgets.FloatSlider(
            value=S.eta,
            min=min(0.1, S.eta),
            max=max(5.0, S.eta),
            step=0.01 if small else 0.1,
            description="eta",
            readout_format=".2f" if small else ".1f",
            style=SLIDER_STYLE,
            layout=widgets.Layout(width="320px"),
        )
        tip(self.eta, "eta, the step size: a gradient step moves each weight by eta times its slope")
        self.buttons = {name: widgets.Button(description=name) for name in ("gradient step", "10 steps", "reset")}
        self.buttons["gradient step"].button_style = "primary"
        tip(self.buttons["gradient step"], "every weight moves against its slope: w <- w - eta x dR/dw")
        tip(self.buttons["10 steps"], "ten gradient steps in a row")
        tip(self.buttons["reset"], "back to the weights the cell started from")
        self.layout_pick = tip(
            widgets.ToggleButtons(options=["wide", "standard"], value=S.layout.style, style={"button_width": "80px"}),
            "wide: all networks in one row; standard: two per row, with larger text on small screens",
        )
        self.layout_pick.observe(self.on_layout, names="value")
        self.layout_box = widgets.HBox(
            [widgets.Label("layout", layout=widgets.Layout(margin="0 8px 0 0")), self.layout_pick],
            layout=widgets.Layout(align_items="center", margin="0 0 0 40px"),
        )

        self.message = widgets.HTML()
        self.readout = widgets.HTML()
        self.image = widgets.Image(format="png", layout=widgets.Layout(max_width="100%", height="auto"))
        self.out = widgets.Output()
        self.status = widgets.VBox([self.readout, self.message], layout=widgets.Layout(margin="4px 0 8px 0"))

        steps_row = widgets.HBox(
            [self.eta, *self.buttons.values(), self.layout_box],
            layout=widgets.Layout(align_items="center", flex_flow="row wrap", margin="0 0 10px 0"),
        )
        core = widgets.VBox([widgets.HTML(text_html(HINTS["core"])), steps_row, self.panel, self.frozen_note])
        # tabs only for what the call asked for; the backward pass stays on while its tab exists
        self.kinds, panels = ["core"], {"core": core}
        if "backprop" in S.views:
            self.kinds.append("backprop")
            panels["backprop"] = self.backprop_tab(widgets)
        if any(v != "backprop" for v in S.views):
            self.kinds.append("views")
            panels["views"] = self.views_tab(widgets)
        if len(self.kinds) == 1:
            self.tabs = None
            self.controls = core
        else:
            self.tabs = widgets.Tab(children=[panels[k] for k in self.kinds])
            for i, kind in enumerate(self.kinds):
                self.tabs.set_title(i, TITLES[kind])
            self.tabs.selected_index = self.kinds.index("backprop") if "backprop" in S.views else 0
            self.controls = self.tabs

        for s in self.sliders:
            s.observe(self.on_slider, names="value")
        self.eta.observe(self.on_eta, names="value")
        self.buttons["gradient step"].on_click(lambda _: self.steps(1))
        self.buttons["10 steps"].on_click(lambda _: self.steps(10))
        self.buttons["reset"].on_click(lambda _: self.reset())

        # the picture first, then the status line, then the controls
        self.picture_box = self.out if self.use_output else self.image
        tail = [] if self.use_output else [self.out]
        self.root = widgets.VBox([self.picture_box, self.status, self.controls, *tail])
        self.root.layout.width = "100%"
        self.arrange(S.layout.style)
        self.widgets = [self.root]
        self.say(f"At the start: R = {S.history[0]:.4f}." if "backprop" not in S.views else self.stage_words())

    # ---------------------------------------------------------------- building

    def arrange(self, style):
        """The picture takes the full width in the wide layout; in the standard one it keeps its own width."""
        self.picture_box.layout.max_width = "100%"
        self.picture_box.layout.width = "100%" if style == "wide" else ""

    def grouped_panel(self, widgets):
        """Up to 30 sliders: one column per unit with weights that move, headed by the unit and its inputs; the
        columns line up in a grid."""
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
        return widgets.GridBox(
            boxes,
            layout=widgets.Layout(grid_template_columns="repeat(auto-fill, minmax(270px, 1fr))", grid_gap="2px 16px"),
        )

    def dropdown_panel(self, widgets):
        """More than 30 sliders: choose one weight, then move it."""
        names = [self.session.names[k] for k in self.index]
        pick = widgets.Dropdown(options=[(nm, i) for i, nm in enumerate(names)], value=0, description="weight")
        tip(pick, "choose the weight to move")
        holder = widgets.HBox([self.sliders[0]])
        pick.observe(lambda ch: setattr(holder, "children", [self.sliders[ch["new"]]]), names="value")
        return widgets.HBox([pick, holder])

    def backprop_tab(self, widgets):
        """Previous and next stage, the stage, the case the derivation follows, and the formula of the stage."""
        S = self.session
        n = len(S.backprop_stages)
        self.bp_prev = tip(widgets.Button(description="previous stage"), "the stage before")
        self.bp_next = tip(
            widgets.Button(description="next stage", button_style="primary"), "the next stage, one layer back"
        )
        self.bp_step = widgets.IntSlider(
            value=S.backprop_step, min=0, max=n - 1, description="stage", style=SLIDER_STYLE
        )
        tip(self.bp_step, "0 is the forward pass; the last stage is the average over the cases")
        letters = [chr(65 + c) if c < 26 else str(c) for c in range(S.data.G)]
        self.bp_case = widgets.Dropdown(
            options=letters,
            value=letters[S.backprop_case],
            description="trace case",
            layout=widgets.Layout(width="190px"),
        )
        tip(self.bp_case, "the group whose case the panel under the picture writes out")
        self.bp_caption = widgets.HTML()
        self.bp_prev.on_click(lambda _: setattr(self.bp_step, "value", max(0, self.bp_step.value - 1)))
        self.bp_next.on_click(lambda _: setattr(self.bp_step, "value", min(n - 1, self.bp_step.value + 1)))
        self.bp_step.observe(self.on_backprop, names="value")
        self.bp_case.observe(self.on_backprop, names="value")
        row = widgets.HBox(
            [self.bp_prev, self.bp_next, self.bp_step, self.bp_case],
            layout=widgets.Layout(align_items="center", flex_flow="row wrap"),
        )
        return widgets.VBox([widgets.HTML(text_html(HINTS["backprop"])), row, self.bp_caption])

    def views_tab(self, widgets):
        """A checkbox per extra panel with a line on what it shows, and the settings of the panels."""
        S = self.session
        names = list(S.names)
        self.view_boxes = {
            v: tip(widgets.Checkbox(value=v in S.views, description=VIEW_LABELS[v], indent=False), VIEW_HELP[v])
            for v in VIEWS
            if v != "backprop"
        }
        rows = [
            widgets.HBox([box, widgets.HTML(f'<span style="color:#666;font-size:12px">{VIEW_HELP[v]}</span>')])
            for v, box in self.view_boxes.items()
        ]
        self.case_pick = tip(
            widgets.Dropdown(options=names, value=names[S.case_weight], description="case slopes of"),
            "the weight of the panel 'each case's slope'",
        )
        a = S.slice_weights[0]
        b = S.slice_weights[1] if len(S.slice_weights) > 1 else None
        self.slice_pick = [
            tip(widgets.Dropdown(options=names, value=names[a], description="slice over"), "the first weight"),
            tip(
                widgets.Dropdown(options=["none"] + names, value="none" if b is None else names[b], description="and"),
                "the second weight, or none for a curve over one weight",
            ),
        ]
        self.beta = tip(
            widgets.FloatSlider(value=S.beta, min=0.0, max=0.99, step=0.01, description="beta"),
            "momentum: the share of the last step that each step keeps",
        )
        self.ahead = tip(
            widgets.IntSlider(value=S.lookahead_steps, min=1, max=200, description="steps ahead"),
            "how many steps the momentum panel looks ahead",
        )
        self.log_box = tip(
            widgets.Checkbox(value=S.log_scale, description="R on a log scale", indent=False),
            "draw R on a logarithmic axis",
        )
        for w in [*self.view_boxes.values(), self.case_pick, *self.slice_pick, self.beta, self.ahead, self.log_box]:
            w.observe(self.on_views, names="value")
        settings = widgets.GridBox(
            [self.case_pick, *self.slice_pick, self.beta, self.ahead],
            layout=widgets.Layout(grid_template_columns="repeat(auto-fill, minmax(300px, 1fr))", grid_gap="2px 16px"),
        )
        return widgets.VBox([widgets.HTML(text_html(HINTS["views"])), *rows, self.log_box, settings])

    # ---------------------------------------------------------------- state

    def say(self, text):
        self.message.value = message_html(text)

    def stage_words(self):
        """The status line for the current stage of the backward pass: where to look in the picture."""
        S = self.session
        return LOOK[S.backprop_stages[S.backprop_step].kind]

    def values(self):
        """The full weight vector: the sliders for the weights that move, the network for the others."""
        w = self.session.weights()
        for k, s in zip(self.index, self.sliders):
            w[k] = s.value
        return w

    def on_slider(self, change=None):
        if self.busy:
            return
        before = self.last_R
        self.session.model.set_weights(self.values())
        self.redraw()
        if change is not None and before is not None:
            name, old, new = change["owner"].description, change["old"], change["new"]
            self.say(f"{name} changed from {old:.2f} to {new:.2f}: {change_words(before, self.last_R)}.")

    def on_eta(self, change):
        self.session.eta = float(change["new"])
        if not self.busy:
            self.redraw()
            eta = self.session.eta
            r = self.session.readouts(eta)
            self.say(f"eta is now {eta:g}: the next step should change R by about {r['next_change']:+.4f}.")

    def on_backprop(self, *_):
        S = self.session
        S.backprop_step = int(self.bp_step.value)
        S.backprop_case = ord(self.bp_case.value) - 65 if len(self.bp_case.value) == 1 else int(self.bp_case.value)
        if not self.busy:
            self.redraw()
            self.say(self.stage_words())

    def on_layout(self, change):
        """Switches between the wide and the standard layout: a new figure."""
        S = self.session
        S.set_layout(change["new"])
        self.arrange(change["new"])
        self.picture = None
        self.redraw()
        where = "all networks in one row" if change["new"] == "wide" else "two networks per row, larger text"
        self.say(f"Layout {change['new']}: {where}.")

    def on_views(self, *_):
        if self.busy:
            return
        S = self.session
        names = list(S.names)
        keep = ("backprop",) if "backprop" in S.views else ()
        S.views = tuple(v for v in VIEWS if v in keep or (v in self.view_boxes and self.view_boxes[v].value))
        S.case_weight = names.index(self.case_pick.value)
        first = names.index(self.slice_pick[0].value)
        second = self.slice_pick[1].value
        S.slice_weights = [first] if second in ("none", self.slice_pick[0].value) else [first, names.index(second)]
        S.beta = float(self.beta.value)
        S.lookahead_steps = int(self.ahead.value)
        S.log_scale = bool(self.log_box.value)
        self.picture = None  # the panels changed: a new figure
        self.redraw()
        shown = [VIEW_LABELS[v] for v in S.views if v != "backprop"]
        self.say("Extra panels: " + (", ".join(shown) if shown else "none") + ".")

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
        before = S.risk(self.values())
        S.step(m, S.eta, self.values())
        after = S.history[-1]
        if m == 1:
            p = S.predictions[-1]
            text = (
                f"One gradient step with eta {S.eta:g}: {change_words(before, after)}. "
                f"Predicted change {p.predicted - p.before:+.4f}, actual {p.actual - p.before:+.4f}."
            )
        else:
            text = f"{m} gradient steps with eta {S.eta:g}: {change_words(before, after)}."
        if after > before:
            text += " R went up: eta may be too large."
        self.say(text)

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
        self.say(f"Back to the start: the weights the cell started from, R = {S.history[0]:.4f}.")

    def redraw(self):
        S = self.session
        with self.out:
            if self.picture is None:
                self.picture = Picture(S, dpi=self.dpi)
            w = self.values()
            self.picture.refresh(w)
            S.fig = self.picture.fig
            self.last_R = self.picture.frame.R
            png = self.picture.png()
            self.readout.value = readout_html(S.readouts(self.eta.value, w))
            if "backprop" in self.kinds:
                n = len(S.backprop_stages) - 1
                text = S.backprop_stages[S.backprop_step].text
                self.bp_caption.value = text_html(f"Stage {S.backprop_step} of {n}. {text[:1].upper()}{text[1:]}.")
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
