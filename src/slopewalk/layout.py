"""Where things go in a panel and how big the figure is: positions of inputs, units and boxes, label places, sizes."""

import numpy as np

BW = 0.27  # half-width of a value box; lines end this far from the centre of the box
TABLE_ROWS = 15  # rows per column of the slope table


class Layout:
    """Positions for a network with n_inputs inputs and the hidden layers `hidden` (a list of unit counts)."""

    def __init__(self, n_inputs, hidden):
        self.n_inputs, self.hidden = n_inputs, hidden
        self.blocks, x = [], 1.75
        for _ in hidden:
            self.blocks.append((x, x + 0.75, x + 1.5))  # x of the z box, of the glyph and of the h box
            x += 2.75
        self.x_a, self.x_sigmoid, self.x_p, self.x_loss = x, x + 0.75, x + 1.5, x + 2.4
        self.y_inputs = np.linspace(1.95, 0.45, n_inputs) if n_inputs > 1 else np.array([1.2])
        self.xlim = (-0.95, self.x_loss + 0.6)
        self.ylim = (-0.05, 3.0)

    @staticmethod
    def unit_ys(m):
        """y of the m units of one layer, top to bottom."""
        return np.linspace(2.2, 0.2, m) if m > 1 else np.array([1.2])

    @staticmethod
    def label_fraction(si, j, ns, nt):
        """Where along a line (0: source, 1: target) its label sits; staggered so that labels into one unit differ."""
        return 0.5 if ns * nt == 1 else 0.22 + 0.56 * (si * nt + j) / (ns * nt - 1)

    @staticmethod
    def show_names(n_weights):
        """Names and values on the lines for small networks, values only above 9 weights."""
        return n_weights <= 9

    def per_row(self):
        return 2 if self.x_loss + 1.55 <= 9 else 1

    def rows(self, n_panels):
        return -(-n_panels // self.per_row())

    def figsize(self, n_panels):
        return (15, 2.9 * self.rows(n_panels) + 4.2)

    @staticmethod
    def table_fontsize(n_weights):
        return 9.5 if n_weights > 15 else 11
