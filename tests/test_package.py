"""The package itself: import without warnings, the version in one place, the public names and signatures."""

import inspect
import pathlib
import re
import subprocess
import sys

import pytest

import slopewalk

ROOT = pathlib.Path(__file__).resolve().parents[1]


IMPORT_CHECK = """
import os, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import slopewalk
package_dir = os.path.dirname(slopewalk.__file__)
ours = [str(w.message) for w in caught if w.filename.startswith(package_dir)]
print(slopewalk.__version__)
print(ours)
"""


def test_import_without_warnings():
    """Importing the package raises no warning from its own code (old versions of other packages may warn)."""
    r = subprocess.run([sys.executable, "-c", IMPORT_CHECK], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    version, ours = r.stdout.strip().splitlines()
    assert version == slopewalk.__version__
    assert ours == "[]", ours


def test_version_matches_pyproject_and_citation():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == slopewalk.__version__
    citation = (ROOT / "CITATION.cff").read_text()
    assert re.search(r"^version: (\S+)", citation, re.M).group(1) == slopewalk.__version__


def test_public_names_and_signatures():
    expected = {"explore", "clusters", "xor_centres", "two_points", "weight_table", "Session"}
    assert set(slopewalk.__all__) >= expected
    for name in slopewalk.__all__:
        assert hasattr(slopewalk, name)
    signature = str(inspect.signature(slopewalk.explore))
    assert signature.startswith("(net, X, y, groups=None, eta=1.0, title=None, *, ")
    assert signature == (
        "(net, X, y, groups=None, eta=1.0, title=None, *, task=None, params=None, views=(), case_slopes=None, "
        "slice_weights=None, beta=0.9, lookahead=30, log_scale=False, layout='auto', width=None, "
        "backprop_step=0, backprop_case=0, theme='auto')"
    )
    assert str(inspect.signature(slopewalk.clusters)) == "(centers, labels, noise=0.0, n_per=50, seed=0)"
    assert str(inspect.signature(slopewalk.xor_centres)) == "(noise=0.0, n_per=50, seed=0)"


def test_help_lists_the_functions_and_their_arguments(capsys):
    """slopewalk.help() prints every public function with its signature, the arguments of explore and the session's
    methods; help("name") prints one in full. It is not exported by "from slopewalk import *"."""
    import inspect

    import slopewalk

    slopewalk.help()
    out = capsys.readouterr().out
    for f in (slopewalk.clusters, slopewalk.xor_centres, slopewalk.two_points, slopewalk.weight_table):
        assert f"{f.__name__}{inspect.signature(f)}" in out
    assert "explore(net, X, y, groups=None, eta=1.0, title=None, *, task=None" in out
    params = [p for p in inspect.signature(slopewalk.explore).parameters if p not in ("net", "X", "y")]
    assert all(p in out for p in params)  # every argument of explore is described
    assert "step(k=1, eta=None, w=None)" in out and "record(path, steps=40" in out and "(self" not in out
    assert all(len(line) <= 104 for line in out.splitlines())
    slopewalk.help("Session.step")
    assert capsys.readouterr().out.startswith("Session.step(k=1, eta=None, w=None)\n\n    k gradient steps")
    slopewalk.help(slopewalk.clusters)
    assert "noise" in capsys.readouterr().out
    with pytest.raises(ValueError, match="'fit' is not one of explore, clusters"):
        slopewalk.help("fit")
    assert "help" not in slopewalk.__all__
