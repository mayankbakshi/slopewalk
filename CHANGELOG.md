# Changelog

## Unreleased

- The picture comes first, with a status line and the controls under it. The controls: one line on what to do, a row with eta, the step buttons and the layout switch, and the sliders in a column per unit. Tabs appear only for what the call asks for (Backprop when `views` includes "backprop", Views for other views), and changing tabs leaves the picture as it is. The status line gives the numbers and one sentence on what the last action did; with the backward pass, it says where to look. Every control has a tooltip. The default layout is `"auto"`: wide where the sliders run in Jupyter or VS Code, standard in Colab and for still pictures.
- Backpropagation step by step: `views="backprop"` (or `backprop_step=k`) draws the backward pass stage by stage, from δ at the output through the slopes of the output weights, ∂ℓ/∂h, δ through the activation (with its tangent) and the slopes of each earlier layer, to the average over the cases. A text panel writes out the numbers for one case (`backprop_case`); the widgets step through the stages; `Session.record_backprop` writes them to a GIF. Every value is computed with the taught formulas and checked against autograd.
- `layout="wide"` and `width`: a figure 27 inches wide with up to four network panels and six lower panels per row. Each lower panel gets a width that suits it, and every row is filled. Narrow tables use short headers.
- The network panels keep the same size and spacing whatever panels are below them.
- New options of `explore`, all keyword-only, so existing calls are unchanged: `task="regression"` (mean squared error, a prediction box instead of the sigmoid and p, the fitted curve or surface in the input panel), `params` (the weights that move; the others are drawn lighter), `views` (extra panels: each case's slope along one weight, R over one or two weights with the path of the steps, R along the next step with its first-order prediction, the cases in the hidden layer, and gradient descent against momentum), `case_slopes`, `slice_weights`, `beta`, `lookahead` and `log_scale`.
- The first-order prediction -eta times the sum of the squared slopes is recorded for every step and shown next to the actual change: in the history panel, in its note, and in a new readout strip of the widgets.
- The starting boundary is drawn dashed once the weights have moved; a network with one input shows p against x1.
- `Session.record(path, steps)` writes a GIF of a run (MP4 when ffmpeg is installed). `Session.predict`, `lookahead`, `case_slopes` and `readouts` give the numbers behind the panels.
- Clear errors for points and networks that slopewalk cannot use: NaN or inf, wrong shapes or lengths, labels other than 0 and 1, `bias=False`, half precision or mixed precision, networks off the CPU. float64 networks work throughout.
- A step that takes a weight past its slider's end widens the slider instead of clipping the weight; sliders start wider around weights beyond ±4; `step()` called in code updates the sliders.
- The widgets keep one figure and send it as a PNG to an image widget: a slider move redraws in about 200 ms instead of about 380 ms for the 13-weight example, without flicker. `SLOPEWALK_DISPLAY=output` shows the picture through `display` instead.
- Label placement is vectorised (the same positions, about 30 times faster) and runs after the layout of the figure; rows and activation glyphs adapt to up to 12 units per layer, and long tables use two taller blocks.
- `sine_points()`: a regression example with one input.

- Renamed from gradviz to slopewalk and split into a package (`src/slopewalk`): model, data, layout, draw, widgets, static, session.
- `explore` returns a `Session` with `save(path)`, `step(k)`, `reset()`, `weights()` and `history`. Nothing else changed in what it draws or computes.
- New `weight_table(net)`: name, layer, unit, input and value of every weight.
- `slopes` no longer touches the `.grad` of the network's parameters, and keeps their `requires_grad` flags.
- Edge labels are placed so that they overlap neither each other nor the boxes: each label tries 13 places along its line and small shifts to either side, and the figure degrades from names and values, to values only, to no labels (the values are then in the table and on the sliders). A test checks every example for overlaps.
- Panel rows grow with the number of units in a layer (up to twice the height) and by a fifth for groups of points.
- The table also lists the value of every weight.
- The slope panel is a table with a title, a header per column (weight, change of R for +0.1, slope dR/dw) and the steepest row highlighted. Above 15 weights it continues in a second block.
- `dist/slopewalk_single.py`: the package in one file, for pasting into a notebook cell (`tools/make_single_file.py`).
- Tests, ruff and GitHub Actions (Python 3.9 with the oldest supported versions, 3.12 with Colab's versions, 3.13 with the newest).
