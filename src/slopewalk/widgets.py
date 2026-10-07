"""The controls of explore: a slider per weight, eta, the step buttons and reset. Needs ipywidgets."""

import matplotlib.pyplot as plt
import numpy as np


class Controller:
    """Builds the widgets for a session and keeps them, the network and the picture in step."""

    def __init__(self, session):
        import ipywidgets as widgets

        S = self.session = session
        self.sliders = [
            widgets.FloatSlider(
                value=v,
                min=-4,
                max=4,
                step=0.001,
                description=S.names[k],
                readout_format=".2f",
                continuous_update=False,
                layout=widgets.Layout(width="230px"),
            )
            for k, v in enumerate(S.w0)
        ]
        panel = self.grouped_panel(widgets) if len(self.sliders) <= 30 else self.dropdown_panel(widgets)
        self.eta = widgets.FloatSlider(value=S.eta, min=0.1, max=5, step=0.1, description="eta", readout_format=".1f")
        self.buttons = {name: widgets.Button(description=name) for name in ("gradient step", "10 steps", "reset")}
        self.out = widgets.Output()
        self.busy = False
        for s in self.sliders:
            s.observe(self.redraw, names="value")
        self.buttons["gradient step"].on_click(lambda _: self.steps(1))
        self.buttons["10 steps"].on_click(lambda _: self.steps(10))
        self.buttons["reset"].on_click(lambda _: self.reset())
        self.widgets = [panel, widgets.HBox([self.eta, *self.buttons.values()]), self.out]

    def grouped_panel(self, widgets):
        """Up to 30 weights: the sliders in one box per unit, with the names of the unit's inputs."""
        M, boxes, k = self.session.model, [], 0
        for layer, L in enumerate(M.layers):
            for j in range(L.out_features):
                head = widgets.HTML(
                    f"<b>{M.unit_title(layer, j)}</b>: weights for {', '.join(M.input_labels(layer))}, bias"
                )
                boxes.append(widgets.VBox([head] + self.sliders[k : k + L.in_features + 1]))
                k += L.in_features + 1
        return widgets.Box(boxes, layout=widgets.Layout(flex_flow="row wrap"))

    def dropdown_panel(self, widgets):
        """More than 30 weights: choose one weight, then move it."""
        pick = widgets.Dropdown(
            options=[(nm, i) for i, nm in enumerate(self.session.names)], value=0, description="weight"
        )
        holder = widgets.HBox([self.sliders[0]])
        pick.observe(lambda ch: setattr(holder, "children", [self.sliders[ch["new"]]]), names="value")
        return widgets.HBox([pick, holder])

    def values(self):
        return [s.value for s in self.sliders]

    def set_values(self, w):
        for s, v in zip(self.sliders, w):
            s.value = float(np.clip(v, s.min, s.max))

    def redraw(self, *_):
        if self.busy:
            return
        with self.out:
            self.out.clear_output(wait=True)
            self.session.draw(self.values())
            plt.show()

    def steps(self, m):
        """m gradient steps from the slider values with the current eta; the sliders then show the new weights."""
        self.busy = True
        w = self.session.step(m, self.eta.value, self.values())
        self.set_values(w)
        self.busy = False
        self.redraw()

    def reset(self):
        """Back to the weights at the start, with an empty history."""
        self.busy = True
        w = self.session.reset()
        for s, v in zip(self.sliders, w):
            s.value = v
        self.busy = False
        self.redraw()

    def show(self):
        from IPython.display import display

        display(*self.widgets)
        self.redraw()
        return self.session
