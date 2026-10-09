"""The colours of the notebook editor, for explore(theme="auto"), when the kernel can read them.

The kernel cannot ask the editor which theme it shows, but on the same machine it can read the editor's settings:
VS Code's color theme (or, when VS Code follows the system, the system's appearance), and JupyterLab's or Notebook
7's theme. Colab does not tell the kernel its theme, and neither does an editor on another machine (a remote kernel):
there the picture is light, and the colours switch or theme="dark" changes it. SLOPEWALK_THEME=dark (or light) in the
environment chooses for every notebook.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# VS Code's own themes, as written in its settings
VSCODE_LIGHT = {
    "Default Light Modern",
    "Default Light+",
    "Visual Studio Light",
    "Default High Contrast Light",
    "Quiet Light",
    "Solarized Light",
}
VSCODE_DARK = {
    "Default Dark Modern",
    "Default Dark+",
    "Visual Studio Dark",
    "Default High Contrast",
    "Abyss",
    "Kimbie Dark",
    "Monokai",
    "Monokai Dimmed",
    "Red",
    "Solarized Dark",
    "Tomorrow Night Blue",
}


def auto_theme():
    """ "light" or "dark" for explore(theme="auto"): SLOPEWALK_THEME if set; light for still pictures (NB_STATIC=1) and
    in Colab; else the theme of VS Code or JupyterLab read from their settings, and light when that fails."""
    forced = os.environ.get("SLOPEWALK_THEME", "").strip().lower()
    if forced in ("light", "dark"):
        return forced
    if os.environ.get("NB_STATIC") == "1" or "google.colab" in sys.modules:
        return "light"
    try:
        found = vscode_theme() if os.environ.get("VSCODE_PID") else jupyterlab_theme()
    except Exception:  # a setting the kernel cannot read is not worth an error: the switch is there
        found = None
    return found or "light"


def read_jsonc(path):
    """A settings file in JSON with comments and trailing commas, as VS Code and JupyterLab write them, or None."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] == '"':  # a string, kept as it is
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(text[i])
            i += 1
    try:
        data = json.loads(re.sub(r",(\s*[}\]])", r"\1", "".join(out)))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def system_theme():
    """ "dark" or "light" from the system's appearance (macOS, Windows, GNOME), or None."""
    try:
        if sys.platform == "darwin":
            done = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True, timeout=2
            )
            return "dark" if done.stdout.strip().lower() == "dark" else "light"
        if sys.platform.startswith("win"):
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            )
            return "light" if winreg.QueryValueEx(key, "AppsUseLightTheme")[0] else "dark"
        done = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if done.returncode == 0:
            return "dark" if "dark" in done.stdout else "light"
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def vscode_folder():
    """VS Code's folder of user data (it holds User/settings.json): from the kernel's environment, which VS Code
    sets, or the usual place."""
    cache = os.environ.get("VSCODE_CODE_CACHE_PATH")  # .../Code/CachedData/<version>
    if cache and Path(cache).parent.name == "CachedData":
        return Path(cache).parent.parent
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Code"
    if sys.platform.startswith("win"):
        return Path(os.environ.get("APPDATA", Path.home())) / "Code"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "Code"


def vscode_theme():
    """The kind of theme VS Code shows. VS Code keeps the kind of the theme it last painted in
    User/globalStorage/storage.json (for its start-up screen), which follows extension themes and the system's
    appearance; without it, its settings: the system's appearance when VS Code follows it, else the name of its
    color theme (dark when none is set, VS Code's default)."""
    folder = vscode_folder()
    try:
        storage = json.loads((folder / "User" / "globalStorage" / "storage.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        storage = {}
    splash = storage.get("windowSplash") if isinstance(storage, dict) else None
    base = storage.get("theme") if isinstance(storage, dict) else None
    base = base or (splash.get("baseTheme") if isinstance(splash, dict) else None)
    if base in ("vs", "hc-light"):
        return "light"
    if base in ("vs-dark", "hc-black"):
        return "dark"
    settings = read_jsonc(folder / "User" / "settings.json") or {}
    if settings.get("window.autoDetectColorScheme"):
        return system_theme()
    return theme_kind(str(settings.get("workbench.colorTheme", "Default Dark Modern")))


def theme_kind(name):
    """ "light" or "dark" for the name of a VS Code color theme: VS Code's own themes, else a word in the name, or
    None."""
    if name in VSCODE_LIGHT:
        return "light"
    if name in VSCODE_DARK:
        return "dark"
    words = name.lower()
    if "light" in words:
        return "light"
    if any(w in words for w in ("dark", "night", "black", "monokai", "dracula")):
        return "dark"
    return None


def jupyterlab_theme():
    """The theme of JupyterLab or Notebook 7 on this machine, from its user settings, or None (the default, light)."""
    try:
        from jupyter_core.paths import jupyter_config_dir

        base = Path(jupyter_config_dir())
    except ImportError:  # jupyter_core comes with Jupyter; without it, its rule: JUPYTER_CONFIG_DIR, else ~/.jupyter
        base = Path(os.environ.get("JUPYTER_CONFIG_DIR") or Path.home() / ".jupyter")
    path = base / "lab" / "user-settings" / "@jupyterlab" / "apputils-extension" / "themes.jupyterlab-settings"
    settings = read_jsonc(path)
    if not settings:
        return None
    if settings.get("adaptive-theme"):
        return system_theme()
    words = str(settings.get("theme", "JupyterLab Light")).lower()
    if "light" in words:
        return "light"
    if "dark" in words or "night" in words:
        return "dark"
    return None
