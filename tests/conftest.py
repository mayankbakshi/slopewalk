import os
import pathlib
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))  # render_examples: the example cases and the reference run


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def light_unless_asked(monkeypatch):
    """explore(theme="auto") reads the editor's theme; the tests run with light colours unless they ask otherwise."""
    monkeypatch.setenv("SLOPEWALK_THEME", "light")
