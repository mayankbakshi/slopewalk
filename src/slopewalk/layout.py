"""Where things go in a panel and how big the figure is: positions of inputs, units and boxes, label places, sizes."""

import numpy as np

BW = 0.27  # half-width of a value box; lines end this far from the centre of the box
TABLE_ROWS = 15  # rows per column of the slope table
LABEL_FRACTIONS = np.linspace(0.14, 0.86, 13)  # where along a line a label may sit
LABEL_SHIFTS = (0.0, 0.8, -0.8, 1.6, -1.6, 2.4, -2.4, 3.2, -3.2, 4.0, -4.0, 4.8, -4.8)  # sideways shifts
LABEL_TOLERANCE = 0.02  # a label may overlap other labels or the boxes by at most this fraction of its area
BOTTOM_PER_ROW = 3  # panels per row below the networks
BOTTOM_HEIGHT = 4.2  # height in inches of one row of those panels
LAYOUTS = ("standard", "wide")
WIDTH = {"standard": 15.0, "wide": 27.0}  # default figure width in inches
WIDE_PER_ROW, WIDE_BOTTOM_PER_ROW = 4, 6  # the most network panels and lower panels per row of the wide layout
NET_GAP, PANEL_GAP = 0.5, 0.8  # inches between network panels, and between the panels below them
OUTPUT_GAP = 0.4  # extra room before several output units (more would make the panel too wide for two in a row)
TEXT_SCALE = {"standard": 1.0, "wide": 1.1}  # text in the network panels: the wide picture is shown smaller
# the wide layout gives each lower panel a base width in inches that suits it (for a row 4.2 inches tall), and the
# width left over in its row goes to the panels that can use it, in these proportions
WIDE_BASE = {
    "plane": 4.6,
    "hidden": 4.6,
    "slice": 4.8,
    "table": 5.0,
    "history": 5.0,
    "line": 5.0,
    "cases": 5.0,
    "momentum": 5.0,
    "backprop": 10.5,
}
HISTORY_MIN = 3.6  # the narrowest R over the steps gets in a full row of the wide layout
WIDE_STRETCH = {"history": 3.0, "table": 0.5, "line": 1.0, "cases": 1.0, "momentum": 1.0, "backprop": 1.0}


class Layout:
    """Positions for a network with n_inputs inputs and the hidden layers `hidden` (a list of unit counts).

    task "binary" ends each panel with a, the sigmoid, p and the loss box; "multiclass" with a score a and a p for
    each of the `outputs` classes, the softmax between them, and the loss box; "regression" with the prediction and
    the loss box. style "standard": a figure 15 inches wide, two network panels per row (one for deep networks) and
    three lower panels per row. "wide": 27 inches, up to four network panels in a row (two for deep networks) and up
    to six lower panels in a row, each as wide as suits it, for wide screens. width (inches) overrides the width of
    either. The network panels keep their size and spacing whatever panels are below them.
    """

    def __init__(self, n_inputs, hidden, task="binary", style="standard", width=None, outputs=1):
        if style not in LAYOUTS:
            raise ValueError(f"slopewalk: layout must be 'standard' or 'wide', not {style!r}.")
        if width is not None and not (isinstance(width, (int, float)) and 6 <= width <= 60):
            raise ValueError(f"slopewalk: width is the figure width in inches, between 6 and 60, not {width!r}.")
        self.n_inputs, self.hidden, self.task, self.style, self.outputs = n_inputs, hidden, task, style, outputs
        self.width = float(width) if width is not None else WIDTH[style]
        self.text_scale = f = TEXT_SCALE[style]
        extra = 1.2 * (f - 1)  # larger text: the loss box moves right, clear of the box before it
        self.blocks, x = [], 1.75
        for _ in hidden:
            self.blocks.append((x, x + 0.75, x + 1.5))  # x of the z box, of the glyph and of the h box
            x += 2.75
        if outputs > 1:  # the lines fan out to several output units: more room for their labels
            x += OUTPUT_GAP
        if task == "regression":
            gap = 1.1 if outputs == 1 else 1.6  # several targets: room for their arrows into the loss box
            self.x_a, self.x_sigmoid, self.x_p, self.x_loss = x, None, None, x + gap + extra
        else:
            self.x_a, self.x_sigmoid, self.x_p, self.x_loss = x, x + 0.75, x + 1.5, x + 2.4 + extra
        self.x_input_value = -0.24 * f  # the right edge of an input's value box, left of its circle
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

    def narrow(self):
        """A network narrow enough for two panels side by side in the standard figure (at most one hidden layer)."""
        return self.x_loss + 1.55 <= 9

    def per_row(self, n_panels=None):
        """Network panels per row: 2 (or 1 for deep networks); the wide layout puts up to 4 (or 2) in a row."""
        if self.style == "wide":
            most = WIDE_PER_ROW if self.narrow() else 2
            return max(1, min(most, n_panels if n_panels else most))
        return 2 if self.narrow() else 1

    def rows(self, n_panels):
        return -(-n_panels // self.per_row(n_panels))

    def row_height(self, with_sd=False, backprop=False):
        """Height of a panel row in inches: taller when a layer has more than 3 units (up to four times the usual,
        for 12 units; the outputs count as a layer), by a fifth when the boxes carry a standard deviation line (groups
        of points), and by 40% when the backward pass adds its boxes under the units."""
        units = min(4.0, max(1.0, max(list(self.hidden) + [self.outputs]) / 3))
        return 2.9 * units * (1.2 if with_sd else 1.0) * (1.4 if backprop else 1.0) * self.text_scale

    def lower_rows(self, panels, n_weights=0):
        """The rows of the panels below the networks: lists of (panel, width), with widths in inches that add up to
        the width of the figure.

        standard: three columns of equal width; the derivation of the backward pass takes a whole row, and the spare
        columns of a short last row stay empty (panel None). wide: up to six columns in a row (the derivation counts
        two), the rows about equally full and every row filled: each panel gets its base width, and the width left
        over goes to the panels that can use it (mostly R over the steps).
        """
        wide = self.style == "wide"
        spans = [(2 if wide else BOTTOM_PER_ROW) if p == "backprop" else 1 for p in panels]
        cap = -(-sum(spans) // -(-sum(spans) // WIDE_BOTTOM_PER_ROW)) if wide else BOTTOM_PER_ROW
        rows, row, used = [], [], 0
        for name, span in zip(panels, spans):
            if row and used + span > cap:
                rows.append(row)
                row, used = [], 0
            row.append((name, span))
            used += span
        rows.append(row)
        if not wide:
            third = self.width / BOTTOM_PER_ROW
            return [[(n, third * k) for n, k in r] + [(None, third)] * (cap - sum(k for _, k in r)) for r in rows]
        return [self.fill([n for n, _ in r], n_weights) for r in rows]

    def fill(self, names, n_weights=0):
        """Widths for one row of the wide layout: base widths, and the rest of the row shared by WIDE_STRETCH."""
        base = [WIDE_BASE[n] for n in names]
        if "table" in names and n_weights > TABLE_ROWS:
            base[names.index("table")] = 7.5  # two blocks of rows side by side
        spare = self.width - sum(base)
        if spare < 0 and "history" in names:  # a full row: R over the steps gives up width first, the table keeps it
            i = names.index("history")
            base[i] -= min(-spare, max(0.0, base[i] - HISTORY_MIN))
            spare = self.width - sum(base)
        if spare <= 0:
            return [(n, b * self.width / sum(base)) for n, b in zip(names, base)]
        stretch = [WIDE_STRETCH.get(n, 0.0) for n in names]
        if sum(stretch) == 0:
            stretch = [1.0] * len(names)
        return [(n, b + spare * k / sum(stretch)) for n, b, k in zip(names, base, stretch)]

    def wspace(self, n, gap):
        """The wspace of a row of n panels with gaps of `gap` inches (wspace is a fraction of the mean panel width)."""
        return 0.0 if n < 2 else n * gap / max(self.width - 0.6 - (n - 1) * gap, 1.0)

    @staticmethod
    def table_fontsize(n_weights):
        return 9.5 if n_weights > 15 else 11


def overlap_area(a, b):
    """The area shared by two boxes (x0, y0, x1, y1)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def overlap_matrix(A, B):
    """The areas shared by every box of A (m x 4) with every box of B (k x 4), as an m x k array."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    w = np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0])
    h = np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1])
    return np.clip(w, 0, None) * np.clip(h, 0, None)


def place_labels(edges, sizes, obstacles, defaults, pad=0.0):
    """Where each line's label goes, so that labels avoid the boxes and each other.

    edges: [((x0, y0), (x1, y1)), ...]; sizes: [(width, height), ...] of the labels; obstacles: boxes
    (x0, y0, x1, y1) to keep clear of; defaults: the preferred fraction along each edge; all in data units.
    Greedy, edge by edge: of the fractions in LABEL_FRACTIONS, on the line or shifted to either side of it by
    multiples of the label's height, the place with the least overlap with the obstacles and the labels placed so
    far wins, with a slight preference for the default fraction on the line.
    Returns (positions, boxes, worst_labels, worst_boxes): the centre of each label, its box, and the largest
    overlap of any label with other labels and with the obstacles, as fractions of the label's own area.
    """
    obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 4)
    fractions = LABEL_FRACTIONS[:, None]
    shifts = np.asarray(LABEL_SHIFTS, dtype=float)[None, :]
    placed = np.zeros((0, 4))
    positions, boxes, worst_labels, worst_boxes = [], [], 0.0, 0.0
    for (p0, p1), (lw, lh), t0 in zip(edges, sizes, defaults):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        nx, ny = -dy / length, dx / length  # the unit normal of the line
        offset = shifts * lh
        cx = (p0[0] + fractions * dx + offset * nx).ravel()
        cy = (p0[1] + fractions * dy + offset * ny).ravel()
        cand = np.stack([cx - lw / 2 - pad, cy - lh / 2 - pad, cx + lw / 2 + pad, cy + lh / 2 + pad], axis=1)
        with_boxes = overlap_matrix(cand, obstacles).sum(1)
        with_labels = overlap_matrix(cand, placed).sum(1)
        score = with_boxes + with_labels + 1e-4 * (np.abs(fractions - t0) + np.abs(offset)).ravel()
        best = int(np.argmin(score))
        placed = np.vstack([placed, cand[best]])
        positions.append((float(cx[best]), float(cy[best])))
        boxes.append(tuple(float(v) for v in cand[best]))
        area = (lw + 2 * pad) * (lh + 2 * pad)
        worst_labels = max(worst_labels, float(with_labels[best]) / area)
        worst_boxes = max(worst_boxes, float(with_boxes[best]) / area)
    return positions, boxes, worst_labels, worst_boxes
