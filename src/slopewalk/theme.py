"""The colours, fonts and sizes of the picture: a light theme and a dark one, and the role of every colour."""

import copy

from matplotlib.colors import LinearSegmentedColormap


class Theme:
    """Colours, fonts and sizes of the picture. Red is class 1 and blue class 0 (with more classes, teal is class 2 and
    magenta class 3); green is the hidden layer; purple is the backward pass; orange marks negative weights. Text in
    boxes is ink: a box carries its meaning in its border and fill. Ratios are WCAG contrast against paper unless said
    otherwise.

    Only the labels on the weight lines use edge_label_size, and only the weight lines use zorder 1.
    """

    paper, ink, muted = "#FFFFFF", "#1F2328", "#59636E"  # ink 15.8:1, muted 6.1:1
    class_colour = {0: "#0969DA", 1: "#CF222E", 2: "#00796B", 3: "#B4237F"}  # 5.2, 5.4, 5.3, 6.0:1: titles, marks
    class_tint = {0: "#E3EFFD", 1: "#FDECEC", 2: "#DFF3F0", 3: "#FBEAF3"}  # box fills; the text on them is ink
    mixed_colour, mixed_tint = "#1F2328", "#EEF1F4"
    neutral_colour, neutral_tint = "#3D444D", "#EEF1F4"  # regression panels
    positive, negative, negative_text = "#1F2328", "#D1600A", "#B04800"  # lines 15.8:1 and 3.9:1, numbers 5.6:1
    line_width, line_slope, line_max = 1.0, 1.3, 7  # a weight's line: line_width + line_slope * |w|, at most line_max
    frozen_alpha, frozen_text, frozen_fill = 0.35, "#59636E", "#F6F8FA"  # weights that do not move (params)
    hidden = "#1A7F37"  # the hidden layer only: the border of the h boxes, the dot and the band on the curves
    arrow, curve, node_edge = "#8C959F", "#59636E", "#8C959F"
    start_colour, prediction_colour = "#59636E", "#8C959F"
    gd_colour, momentum_colour = "#59636E", "#9A6700"  # gradient descent solid, momentum dashed
    value_box = dict(boxstyle="round,pad=0.25", fc="#F6F8FA", ec="#C4CBD3", lw=0.8)
    edge_label_size, glyph_name_size, value_size, node_size, title_size, suptitle_size = 8.5, 8, 9.5, 10, 11.5, 13
    note_size = 11  # notes inside panels, and the lines of the worked example (smaller only if they do not fit)
    panel_title_size, legend_size, letter_size = 12, 9.5, 12  # the panels below the networks
    tick_size, axis_label_size = 10.5, 11
    plane_alpha, value_cmap = 0.4, "viridis"
    table_size, table_leading = 12, 1.45  # the table's largest font; a row is table_leading times the font
    table_edge, table_head, table_highlight = "#D0D7DE", "#EEF1F4", "#FFF4C2"
    backprop_colour, backprop_tint, backprop_size = "#7A45D6", "#F4EFFC", 9  # 5.8:1, and 5.1:1 on its tint
    # surfaces and neutral marks: every colour of the picture lives in this class, so that a theme can change them all
    marker_edge, boundary = "#1F2328", "#1F2328"
    note_text, derivation_text, derivation_muted, point_neutral = "#59636E", "#1F2328", "#8C959F", "#59636E"
    glyph_fill, glyph_edge, node_fill, label_backing = "#FFFFFF", "#C4CBD3", "#FFFFFF", "#FFFFFF"
    reference_line, zero_line, landscape_line, plane_surface, path_colour = (
        "#C4CBD3",
        "#D0D7DE",
        "#8C959F",
        "#8C959F",
        "#1F2328",
    )
    landscape_low = "#59636E"  # R over two weights: low R in this colour, high R in paper
    spine = "#8C959F"  # the frames of the panels below the networks
    control_button, control_selected = "#EEF1F4", "#C4CBD3"  # the controls: buttons, and the chosen toggle button
    # what a stage of the backward pass does not use: boxes and curves recede to 30%, their numbers to 50% so that they
    # stay readable (3.2:1 on light paper, 4.9:1 on dark), lines to 12% (context only: the table keeps every weight)
    faded_alpha, faded_text_alpha, faded_line_alpha = 0.3, 0.5, 0.12

    def plane_colormap(self):
        """p for class 1 over the input plane: class 0 colour, paper at p = 0.5, class 1 colour."""
        return LinearSegmentedColormap.from_list(
            "slopewalk_plane", [self.class_colour[0], self.paper, self.class_colour[1]]
        )

    def landscape_colormap(self):
        return LinearSegmentedColormap.from_list("slopewalk_landscape", [self.landscape_low, self.paper])

    network_sizes = ("edge_label_size", "glyph_name_size", "value_size", "node_size", "title_size", "backprop_size")

    def scaled(self, factor):
        """A copy whose text in the network panels is `factor` times larger; the panels below keep their sizes."""
        out = copy.copy(self)
        for name in self.network_sizes:
            setattr(out, name, round(getattr(self, name) * factor, 2))
        return out


class DarkTheme(Theme):
    """The same picture for a dark notebook theme, in high contrast: light ink on near-black paper, bright class
    colours, and frames and box edges that stand out. Ratios against #0D1117."""

    paper, ink, muted = "#0D1117", "#F0F3F6", "#B1BAC4"  # ink 17.0:1, muted 9.6:1
    class_colour = {0: "#71B7FF", 1: "#FF9492", 2: "#4FD8C8", 3: "#FF8DCA"}  # 8.9, 8.9, 10.8 and 8.9:1
    class_tint = {0: "#0F2742", 1: "#3A1519", 2: "#0E2E2B", 3: "#36162A"}
    mixed_colour, mixed_tint = "#F0F3F6", "#21262D"
    neutral_colour, neutral_tint = "#D1D9E0", "#21262D"
    positive, negative, negative_text = "#E6EDF3", "#FFA657", "#FFA657"  # lines 16.0:1, negative 9.8:1
    frozen_text, frozen_fill = "#9EA7B3", "#161B22"
    hidden = "#56D364"
    arrow, curve, node_edge = "#9EA7B3", "#C9D1D9", "#9EA7B3"
    start_colour, prediction_colour = "#B1BAC4", "#9EA7B3"
    gd_colour, momentum_colour = "#B1BAC4", "#F0B72F"
    value_box = dict(boxstyle="round,pad=0.25", fc="#161B22", ec="#7A828E", lw=0.8)  # edges 4.9:1
    plane_alpha = 0.5
    table_edge, table_head, table_highlight = "#6E7681", "#21262D", "#4D3B00"
    backprop_colour, backprop_tint = "#CB9EFF", "#261A3D"  # 8.9:1, and 7.6:1 on its tint
    marker_edge, boundary = "#0D1117", "#F0F3F6"
    note_text, derivation_text, derivation_muted, point_neutral = "#B1BAC4", "#F0F3F6", "#9EA7B3", "#B1BAC4"
    glyph_fill, glyph_edge, node_fill, label_backing = "#0D1117", "#7A828E", "#0D1117", "#0D1117"
    reference_line, zero_line, landscape_line, plane_surface, path_colour = (
        "#6E7681",
        "#57606A",
        "#8B949E",
        "#8B949E",
        "#F0F3F6",
    )
    landscape_low = "#D1D9E0"
    spine = "#7A828E"
    control_button, control_selected = "#21262D", "#4F5864"  # ink on them 13.7:1 and 6.5:1


THEME = Theme()
DARK_THEME = DarkTheme()
THEMES = {"light": THEME, "dark": DARK_THEME}


def as_theme(theme):
    """A Theme from "light", "dark", a Theme, or None (light)."""
    if theme is None:
        return THEME
    if isinstance(theme, str):
        if theme not in THEMES:
            raise ValueError(f"slopewalk: theme must be 'light' or 'dark', not {theme!r}.")
        return THEMES[theme]
    return theme
