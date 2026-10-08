"""Where things go in a panel and how big the figure is: positions of inputs, units and boxes, label places, sizes."""

import numpy as np

BW = 0.27  # half-width of a value box; lines end this far from the centre of the box
TABLE_ROWS = 15  # rows per column of the slope table
LABEL_FRACTIONS = np.linspace(0.14, 0.86, 13)  # where along a line a label may sit
LABEL_SHIFTS = (0.0, 0.8, -0.8, 1.6, -1.6, 2.4, -2.4, 3.2, -3.2, 4.0, -4.0, 4.8, -4.8)  # sideways shifts
LABEL_TOLERANCE = 0.02  # a label may overlap other labels or the boxes by at most this fraction of its area


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

    def row_height(self, with_sd=False):
        """Height of a panel row in inches: taller when a layer has more than 3 units (up to twice the usual),
        and by a fifth when the boxes carry a standard deviation line (groups of points)."""
        return 2.9 * min(2.0, max(1.0, max(self.hidden, default=1) / 3)) * (1.2 if with_sd else 1.0)

    def figsize(self, n_panels, with_sd=False):
        return (15, self.row_height(with_sd) * self.rows(n_panels) + 4.2)

    @staticmethod
    def table_fontsize(n_weights):
        return 9.5 if n_weights > 15 else 11


def overlap_area(a, b):
    """The area shared by two boxes (x0, y0, x1, y1)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def place_labels(edges, sizes, obstacles, defaults, pad=0.0):
    """Where each line's label goes, so that labels avoid the boxes and each other.

    edges: [((x0, y0), (x1, y1)), ...]; sizes: [(width, height), ...] of the labels; obstacles: boxes
    (x0, y0, x1, y1) to keep clear of; defaults: the preferred fraction along each edge; all in data units.
    Greedy, edge by edge: of the fractions in LABEL_FRACTIONS, on the line or shifted to either side of it by
    the label's height, the place with the least overlap with the obstacles and the labels placed so far wins,
    with a slight preference for the default fraction on the line.
    Returns (positions, boxes, worst_labels, worst_boxes): the centre of each label, its box, and the largest
    overlap of any label with other labels and with the obstacles, as fractions of the label's own area.
    """
    placed, positions, worst_labels, worst_boxes = [], [], 0.0, 0.0
    for (p0, p1), (lw, lh), t0 in zip(edges, sizes, defaults):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        nx, ny = -dy / length, dx / length  # the unit normal of the line
        best = None
        for t in LABEL_FRACTIONS:
            for shift in LABEL_SHIFTS:
                shift = shift * lh
                cx, cy = p0[0] + t * dx + shift * nx, p0[1] + t * dy + shift * ny
                box = (cx - lw / 2 - pad, cy - lh / 2 - pad, cx + lw / 2 + pad, cy + lh / 2 + pad)
                with_boxes = sum(overlap_area(box, o) for o in obstacles)
                with_labels = sum(overlap_area(box, q) for q in placed)
                score = with_boxes + with_labels + 1e-4 * (abs(t - t0) + abs(shift))
                if best is None or score < best[0]:
                    best = (score, (cx, cy), box, with_labels, with_boxes)
        _, centre, box, with_labels, with_boxes = best
        placed.append(box)
        positions.append(centre)
        area = (lw + 2 * pad) * (lh + 2 * pad)
        worst_labels, worst_boxes = max(worst_labels, with_labels / area), max(worst_boxes, with_boxes / area)
    return positions, placed, worst_labels, worst_boxes
