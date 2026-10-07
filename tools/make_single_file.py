"""Builds dist/slopewalk_single.py: the package in one file with the same public API, for pasting into a notebook cell.

    python tools/make_single_file.py            # writes dist/slopewalk_single.py
    python tools/make_single_file.py --check    # exit code 1 when dist/slopewalk_single.py is missing or out of date

The modules are concatenated in dependency order. Their docstrings and imports are removed; the imports of other
modules of the package are dropped, the external imports are collected at the top. Top-level names must be unique
across the modules, since they share one namespace in the single file.
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "slopewalk"
OUT = ROOT / "dist" / "slopewalk_single.py"
ORDER = ["model", "data", "layout", "draw", "static", "widgets", "session"]
STDLIB = {"os", "sys", "re", "math", "copy", "collections", "functools", "itertools", "warnings"}
HEADER = '''"""slopewalk {version}: gradient descent by hand on small PyTorch networks, in one file.

Built by tools/make_single_file.py from the package at https://github.com/mayankbakshi/slopewalk; edit the package,
not this file. Same use as the package:

    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    X, y, groups = clusters([(-1, 1), (1, 1), (-1, -1), (1, -1)], [1, 0, 0, 1])
    explore(net, X, y, groups)
"""
'''
RELATIVE_IMPORT = re.compile(r"^\s*from \.\S* import ")


def split_module(path):
    """(external import statements, body without docstring and imports, top-level names) of one module."""
    src = path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    drop, imports, names = set(), [], set()
    for i, node in enumerate(tree.body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node.value, "value", None), str):
            drop.update(range(node.lineno - 1, node.end_lineno))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            drop.update(range(node.lineno - 1, node.end_lineno))
            if not (isinstance(node, ast.ImportFrom) and node.level > 0):
                imports.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    body = [line for i, line in enumerate(lines) if i not in drop and not RELATIVE_IMPORT.match(line)]
    return imports, "\n".join(body).strip("\n") + "\n", names


def build():
    """The text of the single file."""
    init = (PKG / "__init__.py").read_text()
    version = re.search(r'__version__ = "([^"]+)"', init).group(1)
    exported = re.search(r"__all__ = \[([^\]]*)\]", init).group(1)
    all_imports, sections, seen = [], [], {}
    for name in ORDER:
        imports, body, names = split_module(PKG / f"{name}.py")
        for n in names:
            if n in seen:
                raise SystemExit(f"make_single_file: {n} is defined in both {seen[n]}.py and {name}.py")
            seen[n] = name
        all_imports += imports
        sections.append(f"# {'-' * 24} {name}.py {'-' * 24}\n\n{body}")

    def key(stmt):
        root = stmt.split()[1].split(".")[0]
        return (root not in STDLIB, stmt.startswith("from "), stmt)

    imports = "\n".join(sorted(set(all_imports), key=key))
    parts = [HEADER.format(version=version), imports, "", f'__version__ = "{version}"', "", "", *sections]
    parts.append(f"__all__ = [{exported}]\n")
    return "\n".join(parts)


def main(argv):
    text = build()
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is out of date: run  python tools/make_single_file.py")
            return 1
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return 0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print("wrote", OUT.relative_to(ROOT), f"({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
