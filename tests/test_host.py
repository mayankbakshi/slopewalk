"""theme="auto": the editor's theme, read from VS Code's or JupyterLab's files on this machine; light elsewhere."""

import json
import sys
import types

import pytest

from slopewalk import host


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def vscode(tmp_path, monkeypatch):
    """A VS Code user-data folder in tmp_path, found the way the kernel finds it: through VSCODE_CODE_CACHE_PATH."""
    folder = tmp_path / "Code"
    monkeypatch.setenv("VSCODE_PID", "1")
    monkeypatch.setenv("VSCODE_CODE_CACHE_PATH", str(folder / "CachedData" / "0123abc"))
    monkeypatch.delenv("SLOPEWALK_THEME", raising=False)
    monkeypatch.delenv("NB_STATIC", raising=False)
    monkeypatch.setattr(host, "system_theme", lambda: "dark")
    return folder


def test_settings_with_comments_and_trailing_commas(tmp_path):
    path = write(
        tmp_path / "settings.json",
        '{\n  // the theme\n  "workbench.colorTheme": "Solarized Light", /* a note */\n'
        '  "url": "https://example.org//path",\n  "list": [1, 2,],\n}\n',
    )
    assert host.read_jsonc(path) == {
        "workbench.colorTheme": "Solarized Light",
        "url": "https://example.org//path",
        "list": [1, 2],
    }
    assert host.read_jsonc(tmp_path / "missing.json") is None
    assert host.read_jsonc(write(tmp_path / "bad.json", "{ not json")) is None


@pytest.mark.parametrize(
    "base, kind", [("vs", "light"), ("hc-light", "light"), ("vs-dark", "dark"), ("hc-black", "dark")]
)
def test_vscode_theme_painted_last(vscode, base, kind):
    """VS Code keeps the kind of the theme it shows in globalStorage/storage.json; it wins over the settings."""
    write(vscode / "User" / "globalStorage" / "storage.json", json.dumps({"theme": base}))
    write(vscode / "User" / "settings.json", '{"workbench.colorTheme": "Quiet Light"}')
    assert host.vscode_folder() == vscode
    assert host.vscode_theme() == kind and host.auto_theme() == kind


def test_vscode_theme_from_the_settings(vscode):
    settings = vscode / "User" / "settings.json"
    assert host.vscode_theme() == "dark"  # nothing written: VS Code's default theme is dark
    write(settings, '{"workbench.colorTheme": "Solarized Light"}')
    assert host.vscode_theme() == "light"
    write(settings, '{"workbench.colorTheme": "Abyss"}')
    assert host.vscode_theme() == "dark"
    write(settings, '{"workbench.colorTheme": "Quiet Light", "window.autoDetectColorScheme": true}')
    assert host.vscode_theme() == "dark"  # follows the system, which the fixture makes dark
    write(vscode / "User" / "globalStorage" / "storage.json", json.dumps({"windowSplash": {"baseTheme": "vs"}}))
    assert host.vscode_theme() == "light"


@pytest.mark.parametrize(
    "name, kind",
    [
        ("Default Light Modern", "light"),
        ("Default Dark Modern", "dark"),
        ("Monokai Dimmed", "dark"),
        ("GitHub Light Default", "light"),
        ("Dracula Soft", "dark"),
        ("One Dark Pro", "dark"),
        ("Atom One", None),
    ],
)
def test_theme_names(name, kind):
    assert host.theme_kind(name) == kind


def test_jupyterlab_theme(tmp_path, monkeypatch):
    monkeypatch.setenv("JUPYTER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("VSCODE_PID", raising=False)
    monkeypatch.delenv("SLOPEWALK_THEME", raising=False)
    monkeypatch.delenv("NB_STATIC", raising=False)
    monkeypatch.setattr(host, "system_theme", lambda: "light")
    path = tmp_path / "lab" / "user-settings" / "@jupyterlab" / "apputils-extension" / "themes.jupyterlab-settings"
    assert host.jupyterlab_theme() is None and host.auto_theme() == "light"  # no setting: JupyterLab's light default
    write(path, '{\n  // chosen in the menu\n  "theme": "JupyterLab Dark",\n}')
    assert host.jupyterlab_theme() == "dark" and host.auto_theme() == "dark"
    write(path, '{"theme": "JupyterLab Dark", "adaptive-theme": true}')
    assert host.jupyterlab_theme() == "light"  # follows the system


def test_what_comes_first(vscode, monkeypatch):
    write(vscode / "User" / "globalStorage" / "storage.json", json.dumps({"theme": "vs-dark"}))
    assert host.auto_theme() == "dark"
    monkeypatch.setenv("SLOPEWALK_THEME", "light")  # the environment variable chooses for every notebook
    assert host.auto_theme() == "light"
    monkeypatch.setenv("SLOPEWALK_THEME", "Dark")
    assert host.auto_theme() == "dark"
    monkeypatch.delenv("SLOPEWALK_THEME")
    monkeypatch.setenv("NB_STATIC", "1")  # still pictures for documents stay light
    assert host.auto_theme() == "light"
    monkeypatch.delenv("NB_STATIC")
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))  # Colab does not say
    assert host.auto_theme() == "light"


def test_a_failure_gives_light(vscode, monkeypatch):
    def broken():
        raise RuntimeError("unreadable")

    monkeypatch.setattr(host, "vscode_theme", broken)
    assert host.auto_theme() == "light"


def test_explore_follows_the_editor(monkeypatch):
    import matplotlib.pyplot as plt
    from helpers import make

    from slopewalk import explore, xor_centres
    from slopewalk.draw import DARK_THEME

    monkeypatch.setenv("NB_STATIC", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(host, "auto_theme", lambda: "dark")
    import slopewalk.session

    monkeypatch.setattr(slopewalk.session, "auto_theme", lambda: "dark")
    s = explore(make("tanh3"), *xor_centres())
    assert s.theme_name == "dark" and s.theme.paper == DARK_THEME.paper
    assert explore(make("tanh3"), *xor_centres(), theme="light").theme_name == "light"
    plt.close("all")
