"""Where things go in a panel and how big the figure is: positions of inputs, units and boxes, label places, sizes."""

import numpy as np

BW = 0.27  # half-width of a value box; lines end this far from the centre of the box
TABLE_ROWS = 15  # rows per column of the slope table
LABELS_UP_TO = 15  # networks with more weights carry no values on their lines (numbers="few"); the table has them
LABEL_FRACTIONS = np.linspace(0.14, 0.86, 13)  # where along a line a label may sit
LABEL_SHIFTS = (0.0, 0.8, -0.8, 1.6, -1.6, 2.4, -2.4, 3.2, -3.2, 4.0, -4.0, 4.8, -4.8)  # sideways shifts
LABEL_TOLERANCE = 0.02  # a label may overlap other labels or the boxes by at most this fraction of its area
BOTTOM_PER_ROW = 3  # panels per row below the networks
BOTTOM_HEIGHT = 4.2  # height in inches of one row of those panels
TABLE_SHARE = 0.8  # the share of the table panel's height that its rows may take; the title and margins take the rest
TABLE_TWO_BLOCKS = 7.5  # width in inches of the table panel when its rows run in two blocks (more than TABLE_ROWS)
# a network panel: the units of a layer run from y = 2.2 down to 0.2 (one unit sits at 1.2), the inputs from 1.95 to
# 0.45, and the output row at 1.2, so the network is centred on 1.2; the bias nodes sit above the top unit, and the
# panel's range is symmetric about 1.2, so that every layer sits in the middle of its panel
Y_CENTRE = 1.2
Y_BIAS = 2.6  # the bias nodes, clear of the top unit's box
Y_RANGE = (-0.4, 2.8)  # the panel's y-range: room for the bias nodes above and the curve names below
BIAS_SHIFT = 0.55  # the bias node of a later layer sits this far right of the h column that feeds the layer
ROW_HEIGHT = 3.05  # inches of a panel row for up to 3 units in a layer: the panel's range at about 0.95 in per unit
UNIT_SPACING = 1.44  # inches between the units of a layer that a taller row keeps (0.9 in on the drawn panel)
LAYOUTS = ("standard", "wide")
WIDTH = {"standard": 15.0, "wide": 27.0}  # default figure width in inches
WIDE_PER_ROW, WIDE_BOTTOM_PER_ROW = 4, 6  # the most network panels and lower panels per row of the wide layout
NET_GAP, PANEL_GAP = 0.5, 0.8  # inches between network panels, and between the panels below them
OUTPUT_GAP = 0.3  # extra room before several output units (more would make the panel too wide for two in a row)
TABLE_INCH_PER_POINT = 0.26  # the width a block of the table needs per point of font, with the short headers
ZOOM = 1.5  # a single network panel (one group shown): this much larger than one of two side by side
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
        self.zoom = 1.0  # ZOOM when one network panel is shown (see focus)
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
            loss_gap = 2.4 if outputs == 1 else 2.5  # several classes: p boxes with a ± line next to the loss box
            self.x_a, self.x_sigmoid, self.x_p, self.x_loss = x, x + 0.75, x + 1.5, x + loss_gap + extra
        self.x_input_value = -0.24 * f  # the right edge of an input's value box, left of its circle
        self.y_inputs = np.linspace(1.95, 0.45, n_inputs) if n_inputs > 1 else np.array([Y_CENTRE])
        self.y_bias = Y_BIAS
        self.xlim = (-0.95, self.x_loss + 0.6)
        self.ylim = Y_RANGE

    @staticmethod
    def unit_ys(m):
        """y of the m units of one layer, top to bottom."""
        return np.linspace(2.2, 0.2, m) if m > 1 else np.array([Y_CENTRE])

    @staticmethod
    def label_fraction(si, j, ns, nt):
        """Where along a line (0: source, 1: target) its label sits; staggered so that labels into one unit differ."""
        return 0.5 if ns * nt == 1 else 0.22 + 0.56 * (si * nt + j) / (ns * nt - 1)

    @staticmethod
    def show_names(n_weights):
        """Names and values on the lines for small networks, values only above 9 weights."""
        return n_weights <= 9

    @staticmethod
    def show_values(n_weights):
        """Values on the lines up to LABELS_UP_TO weights; above that the lines carry no numbers (the table has them)
        unless numbers="all"."""
        return n_weights <= LABELS_UP_TO

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

    def focus(self, alone):
        """One network panel shown (alone True): it is drawn ZOOM times larger, text included."""
        self.zoom = ZOOM if alone else 1.0

    def panel_share(self):
        """The share of the row's width that a single shown panel takes: ZOOM times one of two side by side, at most
        the row (a deep network takes the row anyway)."""
        return min(1.0, self.zoom / 2) if self.narrow() else 1.0

    def row_height(self, with_sd=False, backprop=False):
        """Height of a panel row in inches: ROW_HEIGHT for up to 3 units in a layer, then growing so that the units
        of the widest layer stay about 0.9 in apart (the outputs count as a layer), up to four times the usual for 12
        units; a fifth more when the boxes carry a standard deviation line (groups of points), and 40% more when the
        backward pass adds its boxes under the units."""
        m = max(list(self.hidden) + [self.outputs])
        units = min(4.0, max(1.0, UNIT_SPACING * (m - 1) / ROW_HEIGHT))
        scale = (1.2 if with_sd else 1.0) * (1.4 if backprop else 1.0) * self.text_scale * self.zoom
        return ROW_HEIGHT * units * scale

    def table_width(self, n_weights):
        """Width in inches of the table panel: a third of the standard figure, or the wide layout's base width; with
        two blocks of rows (more than TABLE_ROWS weights), TABLE_TWO_BLOCKS in both layouts (at most half the figure),
        so that the font of the table is not squeezed by the width of its blocks."""
        if n_weights > TABLE_ROWS:
            return min(TABLE_TWO_BLOCKS, self.width / 2)
        return WIDE_BASE["table"] if self.style == "wide" else self.width / BOTTOM_PER_ROW

    def table_blocks(self, n_weights):
        """(blocks, width of a block in inches) of the slope table: one block up to TABLE_ROWS weights, else two."""
        blocks = 2 if n_weights > TABLE_ROWS else 1
        panel = self.table_width(n_weights)
        return blocks, (panel - PANEL_GAP) * (1 - 0.03 * (blocks - 1)) / blocks

    def first_row_height(self, n_weights, font, leading):
        """Height in inches of the first row below the networks, which holds the slope table: BOTTOM_HEIGHT, or more
        when the table's rows would not fit at the font its width allows (up to `font` points, a row `leading` times
        that), up to twice BOTTOM_HEIGHT; above that the table's font shrinks."""
        blocks, block_inches = self.table_blocks(n_weights)
        size = max(8.0, min(font, block_inches / TABLE_INCH_PER_POINT))
        rows = min(n_weights, max(TABLE_ROWS, -(-n_weights // 2))) + 1  # the header and the longest block
        needed = rows * size * leading / 72 / TABLE_SHARE
        if needed <= BOTTOM_HEIGHT * 1.05:  # a small shrink of the font is better than a taller row
            return BOTTOM_HEIGHT
        return min(2 * BOTTOM_HEIGHT, needed)

    def lower_rows(self, panels, n_weights=0):
        """The rows of the panels below the networks: lists of (panel, width), with widths in inches that add up to
        the width of the figure.

        standard: three columns of equal width; the derivation of the backward pass takes a whole row, and the spare
        columns of a short last row stay empty (panel None). A table with two blocks of rows (more than TABLE_ROWS
        weights) takes TABLE_TWO_BLOCKS inches of its row, and the other panels of that row share the rest. wide: up
        to six columns in a row (the derivation counts two, and so does a table with two blocks), the rows about
        equally full and every row filled: each panel gets its base width, and the width left over goes to the panels
        that can use it (mostly R over the steps).
        """
        wide = self.style == "wide"
        two_blocks = n_weights > TABLE_ROWS
        spans = []
        for p in panels:
            if p == "backprop":
                spans.append(2 if wide else BOTTOM_PER_ROW)
            elif p == "table" and wide and two_blocks:
                spans.append(2)
            else:
                spans.append(1)
        cap = -(-sum(spans) // -(-sum(spans) // WIDE_BOTTOM_PER_ROW)) if wide else BOTTOM_PER_ROW
        rows, row, used = [], [], 0
        for name, span in zip(panels, spans):
            if row and used + span > cap:
                rows.append(row)
                row, used = [], 0
            row.append((name, span))
            used += span
        rows.append(row)
        if wide:
            return [self.fill([n for n, _ in r], n_weights) for r in rows]
        out = []
        for r in rows:
            column, table = self.width / BOTTOM_PER_ROW, None
            if two_blocks and any(n == "table" for n, _ in r):
                table = self.table_width(n_weights)
                column = (self.width - table) / (cap - 1)  # the other columns of the table's row share the rest
            widths = [(n, table if n == "table" and table else column * k) for n, k in r]
            out.append(widths + [(None, column)] * (cap - sum(k for _, k in r)))
        return out

    def fill(self, names, n_weights=0):
        """Widths for one row of the wide layout: base widths, and the rest of the row shared by WIDE_STRETCH."""
        base = [WIDE_BASE[n] for n in names]
        if "table" in names:
            base[names.index("table")] = self.table_width(n_weights)  # two blocks of rows side by side: wider
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
