"""dist/slopewalk_single.py: up to date, self-contained, and the same numbers and text as the package."""

import importlib.util
import pathlib
import re
import sys

import make_single_file  # tools/ is on sys.path, see conftest.py
import matplotlib.pyplot as plt
import pytest
from render_examples import CASES

import slopewalk

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "slopewalk_single.py"


@pytest.fixture(scope="module")
def single():
    spec = importlib.util.spec_from_file_location("slopewalk_single", DIST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dist_file_is_up_to_date():
    assert DIST.exists(), "run  python tools/make_single_file.py"
    assert DIST.read_text() == make_single_file.build(), "run  python tools/make_single_file.py"


def test_single_file_is_self_contained(monkeypatch):
    src = DIST.read_text()
    assert "from slopewalk" not in src and "import slopewalk" not in src
    assert not re.search(r"^\s*from \.", src, re.M)
    assert not re.search(r"^(import|from) (ipywidgets|IPython)", src, re.M)  # only inside functions
    monkeypatch.setitem(sys.modules, "ipywidgets", None)  # the file loads without ipywidgets
    spec = importlib.util.spec_from_file_location("slopewalk_single_nowidgets", DIST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.explore)


def test_same_api_and_version(single):
    assert single.__version__ == slopewalk.__version__
    assert set(single.__all__) == set(slopewalk.__all__)
    for name in slopewalk.__all__:
        assert name == "__version__" or callable(getattr(single, name))


def texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts] + [ax.get_title() for ax in fig.axes]


@pytest.mark.parametrize("name", list(CASES))
def test_same_numbers_and_text_on_the_examples(name, single, monkeypatch):
    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    build = CASES[name][0]
    results = []
    for mod in (slopewalk, single):
        net, X, y, groups = build()  # the builders seed torch, so both get the same network
        plt.close("all")
        s = mod.explore(net, X, y, groups)
        table = [row[:4] for row in mod.weight_table(net)]
        results.append((s.weights(), s.risk(), s.slopes().tolist(), texts(plt.gcf()), table))
    (w1, R1, g1, t1, tab1), (w2, R2, g2, t2, tab2) = results
    assert w1 == w2 and R1 == R2 and g1 == g2 and tab1 == tab2
    assert t1 == t2
    plt.close("all")
