"""The controls of explore, arranged so that a student can follow them without instructions.

Four numbered tabs: 1 by hand (a slider per weight), 2 gradient steps (eta and the step buttons), 3 backpropagation
(the backward pass stage by stage), 4 more views (extra panels). Each tab opens with what to try. A status bar under
the tabs gives the key numbers and one sentence about the last action. Every control has a tooltip. Needs ipywidgets.
"""

import math
import os

from .picture import LIVE_DPI, Picture
from .views import VIEWS

TAB_TITLES = ("1 · By hand", "2 · Steps", "3 · Backprop", "4 · Views")
HAND, STEPS, BACKPROP, MORE = range(4)
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
    "slice": "contours of R over two weights, with the path of your steps",
    "line": "R along the next step, next to its first-order prediction",
    "hidden": "the cases at their hidden values, with the output unit's boundary",
    "momentum": "the next steps of gradient descent and of momentum, side by side",
}
LOOK = {  # where to look in the picture at each stage of the backward pass
    "forward": "The picture shows the forward pass. Press next ▶ to start at the output.",
    "output": "Look under a: the purple box is δa = p - y of each case.",
    "weights": "Look at the purple lines: each number is δ of the unit times the input of its line.",
    "to_h": "Look at the purple arrows: δ travels back along the weights to each h.",
    "delta": "Look at the purple tangents on the curves: their slope multiplies ∂ℓ/∂h to give δ.",
    "average": "The average of these slopes over the cases is the slope column of the table: what R.backward() gives.",
}
GUIDES = {
    HAND: "<b>Try this.</b> Drag one slider: it changes that weight in the network itself, and the picture "
    "follows. Watch R in the status bar and the loss of each case. Which weight lowers R most for the same "
    "change? The slope column of the table answers before you try.",
    STEPS: "<b>Try this.</b> Press <b>gradient step</b>: every weight moves against its slope, by eta times the "
    "slope. The status bar compares the predicted change of R with the actual one. Then try a large eta, such "
    "as 5, and watch R.",
    BACKPROP: "<b>Try this.</b> Press <b>next ▶</b> to see how the slopes are computed, from the output back to "
    "the first layer. Purple boxes hold the backward values; the panel under the picture writes out the numbers "
    "for one case. Choose the case with <b>trace case</b>.",
    MORE: "<b>More pictures.</b> Tick a box to add a panel under the network. Untick it to remove it.",
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


def message_html(text):
    return f'<div style="font-size:14px;padding:4px 2px;color:#222">{text}</div>'


def guide_html(text):
    return (
        '<div style="font-size:13px;line-height:1.4;background:#F4F8FF;border-left:4px solid #3B6FD8;'
        f'padding:6px 10px;margin:2px 0 8px 0">{text}</div>'
    )


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
                layout=widgets.Layout(width="230px"),
            )
            self.sliders.append(tip(slider, f"{S.names[k]}: drag to change this weight in the network"))
        self.panel = self.grouped_panel(widgets) if len(self.sliders) <= 30 else self.dropdown_panel(widgets)
        frozen = [n for k, n in enumerate(S.names) if k not in set(self.index)]
        self.frozen_note = widgets.HTML(
            f"<i>fixed (drawn lighter, no slider): {name_ranges(frozen)}</i>" if frozen else ""
        )

        small = S.eta < 0.1
        self.eta = widgets.FloatSlider(
            value=S.eta,
            min=min(0.1, S.eta),
            max=max(5.0, S.eta),
            step=0.01 if small else 0.1,
            description="eta",
            readout_format=".2f" if small else ".1f",
        )
        tip(self.eta, "eta, the step size: a gradient step moves each weight by eta times its slope")
        self.buttons = {name: widgets.Button(description=name) for name in ("gradient step", "10 steps", "reset")}
        self.buttons["gradient step"].button_style = "primary"
        tip(self.buttons["gradient step"], "every weight moves against its slope: w <- w - eta x dR/dw")
        tip(self.buttons["10 steps"], "ten gradient steps in a row")
        tip(self.buttons["reset"], "back to the weights the cell started from")

        self.message = widgets.HTML()
        self.readout = widgets.HTML()
        self.image = widgets.Image(format="png", layout=widgets.Layout(max_width="100%", height="auto"))
        self.out = widgets.Output()
        self.status = widgets.VBox([self.message, self.readout])

        self.tabs = widgets.Tab(
            children=[
                widgets.VBox([widgets.HTML(guide_html(GUIDES[HAND])), self.panel, self.frozen_note]),
                widgets.VBox(
                    [
                        widgets.HTML(guide_html(GUIDES[STEPS])),
                        self.eta,
                        widgets.HBox(list(self.buttons.values()), layout=widgets.Layout(flex_flow="row wrap")),
                    ]
                ),
                self.backprop_tab(widgets),
                self.views_tab(widgets),
            ]
        )
        for i, title in enumerate(TAB_TITLES):
            self.tabs.set_title(i, title)
        self.tabs.selected_index = BACKPROP if "backprop" in S.views else HAND
        self.tabs.observe(self.on_tab, names="selected_index")

        for s in self.sliders:
            s.observe(self.on_slider, names="value")
        self.eta.observe(self.on_eta, names="value")
        self.buttons["gradient step"].on_click(lambda _: self.steps(1))
        self.buttons["10 steps"].on_click(lambda _: self.steps(10))
        self.buttons["reset"].on_click(lambda _: self.reset())

        # one box that holds the controls and the picture; arrange() puts them in a band (wide) or a stack (standard)
        self.band = widgets.HBox(layout=widgets.Layout(align_items="flex-start", width="100%"))
        self.picture_box = self.out if self.use_output else self.image
        self.root = widgets.VBox(layout=widgets.Layout(width="100%"))
        self.arrange(S.layout.style)
        self.widgets = [self.root]
        first = "Start with tab 1: drag a slider." if "backprop" not in S.views else self.stage_words()
        self.say(first)

    # ---------------------------------------------------------------- building

    def arrange(self, style):
        """Wide: a band across the top with the tabs on the left and the status on the right, and the picture under
        it at the full width. Standard: the tabs, the status and the picture one above the other."""
        tail = [] if self.use_output else [self.out]
        self.tabs.layout.flex, self.tabs.layout.min_width = ("1 1 auto", "0") if style == "wide" else ("", "")
        self.status.layout.width = "380px" if style == "wide" else ""
        self.status.layout.min_width = "380px" if style == "wide" else ""
        self.status.layout.margin = "0 0 0 12px" if style == "wide" else ""
        self.picture_box.layout.max_width = "100%"
        self.picture_box.layout.width = "100%" if style == "wide" else ""
        if style == "wide":
            self.band.children = [self.tabs, self.status]
            self.root.children = [self.band, self.picture_box, *tail]
        else:
            self.band.children = []
            self.root.children = [self.tabs, self.status, self.picture_box, *tail]

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
        tip(pick, "choose the weight to move")
        holder = widgets.HBox([self.sliders[0]])
        pick.observe(lambda ch: setattr(holder, "children", [self.sliders[ch["new"]]]), names="value")
        return widgets.HBox([pick, holder])

    def backprop_tab(self, widgets):
        """Back and next buttons, the stage, the case the derivation follows, and the formula of the stage."""
        S = self.session
        n = len(S.backprop_stages)
        self.bp_prev = tip(widgets.Button(description="◀ back"), "the stage before")
        self.bp_next = tip(
            widgets.Button(description="next ▶", button_style="primary"), "the next stage, one layer back"
        )
        self.bp_step = widgets.IntSlider(value=S.backprop_step, min=0, max=n - 1, description="stage")
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
        buttons = widgets.HBox([self.bp_prev, self.bp_next, self.bp_case], layout=widgets.Layout(flex_flow="row wrap"))
        return widgets.VBox([widgets.HTML(guide_html(GUIDES[BACKPROP])), buttons, self.bp_step, self.bp_caption])

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
        self.case_pick = widgets.Dropdown(options=names, value=names[S.case_weight], description="case slopes of")
        tip(self.case_pick, "the weight of the panel 'each case's slope'")
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
        self.layout_pick = tip(
            widgets.ToggleButtons(options=["wide", "standard"], value=S.layout.style, description="layout"),
            "wide: the controls beside the picture; standard: the controls above it, with larger text on small screens",
        )
        self.layout_pick.observe(self.on_layout, names="value")
        for w in [*self.view_boxes.values(), self.case_pick, *self.slice_pick, self.beta, self.ahead, self.log_box]:
            w.observe(self.on_views, names="value")
        return widgets.VBox(
            [
                widgets.HTML(guide_html(GUIDES[MORE])),
                self.layout_pick,
                *rows,
                self.log_box,
                widgets.HTML("<b>Settings</b>"),
                self.case_pick,
                *self.slice_pick,
                self.beta,
                self.ahead,
            ]
        )

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
            name = change["owner"].description
            self.say(
                f"You set {name} from {change['old']:.2f} to {change['new']:.2f}: {change_words(before, self.last_R)}."
            )

    def on_eta(self, change):
        self.session.eta = float(change["new"])
        if not self.busy:
            self.redraw()
            r = self.session.readouts(self.session.eta)
            self.say(
                f"eta is now {self.session.eta:g}: the next step should change R by about {r['next_change']:+.4f}."
            )

    def on_backprop(self, *_):
        S = self.session
        S.backprop_step = int(self.bp_step.value)
        S.backprop_case = ord(self.bp_case.value) - 65 if len(self.bp_case.value) == 1 else int(self.bp_case.value)
        if not self.busy:
            self.redraw()
            self.say(self.stage_words())

    def on_tab(self, change):
        """Tab 3 turns the backward pass on; leaving it turns it off."""
        S = self.session
        on = change["new"] == BACKPROP
        if on == ("backprop" in S.views):
            return
        S.views = tuple(v for v in VIEWS if (v == "backprop" and on) or (v != "backprop" and v in S.views))
        self.picture = None
        self.redraw()
        self.say(self.stage_words() if on else "The backward pass is off. Tab 3 turns it on again.")

    def on_layout(self, change):
        """Switches between the wide and the standard layout: a new figure and a new arrangement."""
        S = self.session
        S.set_layout(change["new"])
        self.arrange(change["new"])
        self.picture = None
        self.redraw()
        where = "beside" if change["new"] == "wide" else "above"
        self.say(f"Layout: {change['new']}, with the controls {where} the picture.")

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
            n = len(S.backprop_stages) - 1
            stage = S.backprop_stages[S.backprop_step]
            self.bp_caption.value = (
                f'<div style="font-size:14px;padding:4px 0"><b>Stage {S.backprop_step} of {n}.</b> {stage.text}</div>'
            )
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
