# Changelog

## Unreleased

- Renamed from gradviz to slopewalk and split into a package (`src/slopewalk`): model, data, layout, draw, widgets, static, session.
- `explore` returns a `Session` with `save(path)`, `step(k)`, `reset()`, `weights()` and `history`. Nothing else changed in what it draws or computes.
- New `weight_table(net)`: name, layer, unit, input and value of every weight.
- `slopes` no longer touches the `.grad` of the network's parameters, and keeps their `requires_grad` flags.
- The slope panel is a table with a title, a header per column (weight, change of R for +0.1, slope dR/dw) and the steepest row highlighted. Above 15 weights it continues in a second block.
- `dist/slopewalk_single.py`: the package in one file, for pasting into a notebook cell (`tools/make_single_file.py`).
- Tests, ruff and GitHub Actions (Python 3.9 with the oldest supported versions, 3.12 with Colab's versions, 3.13 with the newest).
