"""The package itself: import without warnings, the version in one place, the public names and signatures."""

import inspect
import pathlib
import re
import subprocess
import sys

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
    assert set(slopewalk.__all__) >= {"explore", "clusters", "xor_centres", "two_points", "weight_table", "Session"}
    for name in slopewalk.__all__:
        assert hasattr(slopewalk, name)
    assert str(inspect.signature(slopewalk.explore)) == "(net, X, y, groups=None, eta=1.0, title=None)"
    assert str(inspect.signature(slopewalk.clusters)) == "(centers, labels, noise=0.0, n_per=50, seed=0)"
    assert str(inspect.signature(slopewalk.xor_centres)) == "(noise=0.0, n_per=50, seed=0)"
