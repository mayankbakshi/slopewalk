"""The controls of explore. The picture comes first; under it a status block (R and the last step's change, large,
then the other numbers and one sentence on the last action), then the controls: one row with eta and the step
buttons, the sliders in a column per unit, and last the layout of the picture. Tabs appear only for what the explore
call asks for: "Backprop" when the views include "backprop", "Views" when other views are asked for. Changing tabs
leaves the picture as it is. Needs ipywidgets.
"""

import math
import os
from contextlib import contextmanager

from .picture import LIVE_DPI, Picture
from .views import VIEWS

# The look of the control area in one place. Text takes the host's colour, so light and dark themes both work;
# quiet text is the same colour at a lower opacity. The two colours of the last step's change keep at least 3.9:1 on
# white and on dark backgrounds (#1E1E1E, #111111), enough for the large text they are used for.
UI = {
    "quiet": "opacity:0.72",
    "text": "font-size:14px",
    "hint": "font-size:13px",
    "small": "font-size:12px",
    "value": "font-size:15px",
    "big": "font-size:22px;font-weight:600",
    "numbers": "font-variant-numeric:tabular-nums",
    "down": "#2E8B57",
    "up": "#E53935",
    "short": "250px",  # weight sliders and drop-down lists
    "long": "320px",  # eta, stage and the settings of the views
    "label": "40px",  # the label column of the weight sliders, eta and stage
    "setting_label": "100px",  # the label column of the other controls
    "column": "270px",  # the narrowest column of the slider grid
    "gap": "16px",
}
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
    "core": "Press gradient step to move every weight against its slope. Drag a slider to change one weight.",
    "backprop": "Step from the output back to the first layer. The text panel below the networks follows one case.",
    "views": "Tick a box to add a view under the network.",
}
BUTTONS = ("gradient step", "10 steps", "back one step", "reset")


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


def span(text, *styles):
    return f'<span style="{";".join(UI[s] for s in styles)}">{text}</span>'


def readout_html(r, action=None):
    """The status numbers: a row of labels, a row of values and a row of notes. R and the change of R in the last
    action (a step, ten steps, back one step or a slider move; action: (change, note)) are large; the other numbers
    are smaller. The columns stay the same before and after the first action."""
    if action is None:
        change, note = span("none yet", "value", "quiet"), ""
    else:
        actual, note = action
        colour = UI["up"] if actual > 0 else UI["down"]
        change = f'<span style="{UI["big"]};color:{colour}">{actual:+.4f}</span>'
    cells = [
        ("R", span(f"{r['R']:.4f}", "big"), ""),
        ("change of R in the last action", change, note),
        (f"next step, eta {r['eta']:g}: predicted change", span(f"{r['next_change']:+.4f}", "value"), ""),
        ("size of the gradient", span(f"{r['grad_norm']:.4f}", "value"), ""),
    ]
    if r["accuracy"] is not None:
        cells.append(("accuracy", span(f"{r['accuracy']:.3f}", "value"), ""))
    pad = f"padding:0 32px 0 0;text-align:left;vertical-align:bottom;{UI['numbers']}"
    rows = [
        "".join(f'<td style="{pad}">{span(label, "small", "quiet")}</td>' for label, _, _ in cells),
        "".join(f'<td style="{pad}">{value}</td>' for _, value, _ in cells),
        "".join(f'<td style="{pad}">{span(note, "small", "quiet") if note else ""}</td>' for _, _, note in cells),
    ]
    return '<table style="border-collapse:collapse">' + "".join(f"<tr>{row}</tr>" for row in rows) + "</table>"


def message_html(text):
    return f'<div style="{UI["text"]};margin:8px 0 2px 0;min-height:1.4em">{text}</div>'


def text_html(text):
    return f'<div style="{UI["hint"]};{UI["quiet"]};margin:2px 0 8px 0">{text}</div>'


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
        self.busy = False  # True while the sliders are set from code: their observers then do nothing
        self.depth = 0  # how many actions are running; the buttons are disabled while it is above 0
        self.picture = None
        self.last_R = None
        self.action = None  # the change of R in the last action and a note on it, for the status
        self.known = len(S.history)  # the length of the history when the controls last looked
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
                style={"description_width": UI["label"]},
                layout=widgets.Layout(width=UI["short"]),
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
            continuous_update=False,
            style={"description_width": UI["label"]},
            layout=widgets.Layout(width=UI["long"]),
        )
        tip(self.eta, "eta, the step size: a gradient step moves each weight by eta times its slope")
        self.buttons = {name: widgets.Button(description=name) for name in BUTTONS}
        self.buttons["gradient step"].button_style = "primary"
        tip(self.buttons["gradient step"], "every weight moves against its slope: w <- w - eta x dR/dw")
        tip(self.buttons["10 steps"], "ten gradient steps in a row")
        tip(self.buttons["back one step"], "undo the last gradient step, and any slider move made after it")
        tip(self.buttons["reset"], "back to the weights the cell started from")
        self.layout_pick = tip(
            widgets.ToggleButtons(options=["wide", "standard"], value=S.layout.style, style={"button_width": "80px"}),
            "wide: all networks in one row; standard: two per row, with larger text on small screens",
        )
        self.layout_pick.observe(self.on_layout, names="value")
        self.layout_row = widgets.HBox(
            [
                widgets.HTML(
                    span("layout of the picture", "small", "quiet"), layout=widgets.Layout(margin="0 12px 0 0")
                ),
                self.layout_pick,
            ],
            layout=widgets.Layout(align_items="center", margin="14px 0 0 0"),
        )

        self.message = widgets.HTML()
        self.readout = widgets.HTML()
        self.image = widgets.Image(format="png", layout=widgets.Layout(max_width="100%", height="auto"))
        self.out = widgets.Output()
        self.status = widgets.VBox([self.readout, self.message], layout=widgets.Layout(margin="6px 0 8px 0"))

        self.button_row = widgets.HBox(list(self.buttons.values()), layout=widgets.Layout(flex_flow="row nowrap"))
        steps_row = widgets.HBox(  # in a narrow window the buttons move below eta together
            [self.eta, self.button_row],
            layout=widgets.Layout(align_items="center", flex_flow="row wrap", margin="0 0 12px 0"),
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
        self.buttons["back one step"].on_click(lambda _: self.step_back())
        self.buttons["reset"].on_click(lambda _: self.reset())

        # the picture first, then the status, the controls, and last the layout of the picture
        self.picture_box = self.out if self.use_output else self.image
        tail = [] if self.use_output else [self.out]
        self.root = widgets.VBox([self.picture_box, self.status, self.controls, self.layout_row, *tail])
        self.root.layout.width = "100%"
        self.arrange(S.layout.style)
        self.widgets = [self.root]
        self.update_buttons()
        self.say(self.stage_words() if "backprop" in S.views else "")

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
        columns = f"repeat(auto-fill, minmax({UI['column']}, 1fr))"
        return widgets.GridBox(boxes, layout=widgets.Layout(grid_template_columns=columns, grid_gap=f"2px {UI['gap']}"))

    def dropdown_panel(self, widgets):
        """More than 30 sliders: choose one weight, then move it."""
        names = [self.session.names[k] for k in self.index]
        pick = widgets.Dropdown(
            options=[(nm, i) for i, nm in enumerate(names)],
            value=0,
            description="weight",
            style={"description_width": UI["label"]},
            layout=widgets.Layout(width=UI["short"]),
        )
        tip(pick, "choose the weight to move")
        holder = widgets.HBox([self.sliders[0]], layout=widgets.Layout(margin=f"0 0 0 {UI['gap']}"))
        pick.observe(lambda ch: setattr(holder, "children", [self.sliders[ch["new"]]]), names="value")
        return widgets.HBox([pick, holder])

    def backprop_tab(self, widgets):
        """Previous and next stage, the stage and the case the text panel below the networks follows."""
        S = self.session
        n = len(S.backprop_stages)
        self.bp_prev = tip(widgets.Button(description="previous stage"), "the stage before")
        self.bp_next = tip(
            widgets.Button(description="next stage", button_style="primary"), "the next stage, one layer back"
        )
        self.bp_step = widgets.IntSlider(
            value=S.backprop_step,
            min=0,
            max=n - 1,
            description="stage",
            continuous_update=False,
            style={"description_width": UI["label"]},
            layout=widgets.Layout(width=UI["long"]),
        )
        tip(self.bp_step, "0 is the forward pass; the last stage is the average over the cases")
        letters = [chr(65 + c) if c < 26 else str(c) for c in range(S.data.G)]
        self.bp_case = widgets.Dropdown(
            options=letters,
            value=letters[S.backprop_case],
            description="trace case",
            style={"description_width": "70px"},
            layout=widgets.Layout(width="190px"),
        )
        tip(self.bp_case, "the group whose case the text panel below the networks follows")
        self.bp_prev.on_click(lambda _: setattr(self.bp_step, "value", max(0, self.bp_step.value - 1)))
        self.bp_next.on_click(lambda _: setattr(self.bp_step, "value", min(n - 1, self.bp_step.value + 1)))
        self.bp_step.observe(self.on_backprop, names="value")
        self.bp_case.observe(self.on_backprop, names="value")
        row = widgets.HBox(
            [self.bp_prev, self.bp_next, self.bp_step, self.bp_case],
            layout=widgets.Layout(align_items="center", flex_flow="row wrap"),
        )
        return widgets.VBox([widgets.HTML(text_html(HINTS["backprop"])), row])

    def views_tab(self, widgets):
        """A checkbox per view with a line on what it shows, and the settings of the views."""
        S = self.session
        names = list(S.names)
        self.view_boxes = {
            v: tip(
                widgets.Checkbox(
                    value=v in S.views,
                    description=VIEW_LABELS[v],
                    indent=False,
                    layout=widgets.Layout(width="200px"),
                ),
                VIEW_HELP[v],
            )
            for v in VIEWS
            if v != "backprop"
        }

        def setting(label=UI["setting_label"], width=UI["long"]):  # each setting with its own layout
            return dict(style={"description_width": label}, layout=widgets.Layout(width=width))

        self.case_pick = tip(
            widgets.Dropdown(options=names, value=names[S.case_weight], description="case slopes of", **setting()),
            "the weight of the view 'each case's slope'",
        )
        a = S.slice_weights[0]
        b = S.slice_weights[1] if len(S.slice_weights) > 1 else None
        self.slice_pick = [
            tip(
                widgets.Dropdown(options=names, value=names[a], description="slice over", **setting(width=UI["short"])),
                "the first weight",
            ),
            tip(
                widgets.Dropdown(
                    options=["none"] + names,
                    value="none" if b is None else names[b],
                    description="and",
                    **setting(label=UI["label"], width="190px"),
                ),
                "the second weight, or none for a curve over one weight",
            ),
        ]
        self.beta = tip(
            widgets.FloatSlider(
                value=S.beta, min=0.0, max=0.99, step=0.01, description="beta", continuous_update=False, **setting()
            ),
            "momentum: the share of the last step that each step keeps",
        )
        self.ahead = tip(
            widgets.IntSlider(
                value=S.lookahead_steps,
                min=1,
                max=200,
                description="steps ahead",
                continuous_update=False,
                **setting(),
            ),
            "how many steps the momentum view looks ahead",
        )
        self.log_box = tip(
            widgets.Checkbox(value=S.log_scale, description="R on a log scale", indent=False),
            "draw R on a logarithmic axis",
        )
        for w in [*self.view_boxes.values(), self.case_pick, *self.slice_pick, self.beta, self.ahead, self.log_box]:
            w.observe(self.on_views, names="value")
        self.settings = {"cases": [self.case_pick], "slice": self.slice_pick, "momentum": [self.beta, self.ahead]}
        rows = [  # each view's checkbox, what it shows, and its settings in the same row
            widgets.HBox(
                [box, widgets.HTML(span(VIEW_HELP[v], "small", "quiet"), layout=widgets.Layout(width="460px"))]
                + self.settings.get(v, []),
                layout=widgets.Layout(align_items="center", flex_flow="row wrap"),
            )
            for v, box in self.view_boxes.items()
        ]
        self.update_settings()
        return widgets.VBox([widgets.HTML(text_html(HINTS["views"])), *rows, self.log_box])

    def update_settings(self):
        """The settings of a view can be changed while the view is on."""
        for v, controls in self.settings.items():
            for w in controls:
                w.disabled = not self.view_boxes[v].value

    # ---------------------------------------------------------------- state

    def say(self, text):
        self.message.value = message_html(text)

    def stage_words(self):
        """The message for the current stage of the backward pass: where to look in the picture."""
        S = self.session
        return LOOK[S.backprop_stages[S.backprop_step].kind]

    def action_buttons(self):
        return [*self.buttons.values()] + ([self.bp_prev, self.bp_next] if "backprop" in self.kinds else [])

    def update_buttons(self):
        """Enables the buttons, except back one step with no step to undo and the stage buttons at the ends."""
        S = self.session
        for b in self.action_buttons():
            b.disabled = False
        self.buttons["back one step"].disabled = not S.undo
        if "backprop" in self.kinds:
            self.bp_prev.disabled = S.backprop_step == 0
            self.bp_next.disabled = S.backprop_step == len(S.backprop_stages) - 1

    @contextmanager
    def working(self):
        """Disables the buttons while an action runs and the picture is drawn again, so that clicks do not queue up;
        enables them afterwards, also when the action fails."""
        self.depth += 1
        if self.depth == 1:
            for b in self.action_buttons():
                b.disabled = True
        try:
            yield
        finally:
            self.depth -= 1
            if self.depth == 0:
                self.update_buttons()

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
        S = self.session
        if change is not None:
            S.last_moved = self.index[self.sliders.index(change["owner"])]
            if before is not None:
                self.action = (S.risk(self.values()) - before, f"{change['owner'].description} moved by hand")
        S.model.set_weights(self.values())
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
        self.update_settings()
        shown = [VIEW_LABELS[v] for v in S.views if v != "backprop"]
        self.say("Views: " + (", ".join(shown) if shown else "none") + ".")

    def note_action(self):
        """The last action, read from the history: k new steps, a step undone, or a fresh start."""
        S = self.session
        n = len(S.history)
        if n == 1 and not S.predictions:
            self.action = None
        elif n > self.known and self.last_R is not None:
            k = n - self.known
            predicted = sum(p.predicted - p.before for p in S.predictions[-k:])
            what = "one step" if k == 1 else f"{k} steps"
            self.action = (S.history[-1] - self.last_R, f"{what}, predicted {predicted:+.4f}")
        elif n < self.known and self.last_R is not None:
            self.action = (S.risk() - self.last_R, "back one step")
        self.known = n

    def sync(self):
        """After steps, back one step, reset or record, also in code: the status notes the action, the sliders show
        the network's weights, then one redraw."""
        self.note_action()
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
        with self.working():
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

    def step_back(self):
        """Undoes the last gradient step; the session then updates the sliders."""
        S = self.session
        with self.working():
            before = S.risk(self.values())
            if S.step_back() is not None:
                self.say(f"Back one step: {change_words(before, S.risk())}.")

    def reset(self):
        """Back to the weights at the start, with no steps taken, and the sliders' starting ranges."""
        S = self.session
        with self.working():
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
        with self.working(), self.out:
            if self.picture is None:
                self.picture = Picture(S, dpi=self.dpi)
            w = self.values()
            self.picture.refresh(w)
            S.fig = self.picture.fig
            self.last_R = self.picture.frame.R
            png = self.picture.png()
            self.readout.value = readout_html(S.readouts(self.eta.value, w), self.action)
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
