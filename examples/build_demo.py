"""Builds slopewalk_demo.ipynb: six uses of slopewalk. Run from this folder: python build_demo.py"""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# slopewalk: gradient descent by hand

Each cell below draws a small network once for each point (or cluster of points), with sliders for every weight and a button for a gradient step. Colab: run the install cell first. Jupyter or VS Code: `pip install git+https://github.com/mayankbakshi/slopewalk` once.
""")
code(r"""
import importlib.util
if importlib.util.find_spec("slopewalk") is None:      # Colab: installs the package (about 20 seconds)
    %pip install -q git+https://github.com/mayankbakshi/slopewalk
""")
code(r"""
import torch
import torch.nn as nn
from slopewalk import explore, clusters
""")
md(r"""
## 1. Two points, no hidden layer

Logistic regression, $a = w_1 x_1 + w_2 x_2 + w_3$. Start: $B$ on the class 1 side. Which weight lowers $R$ most? Then press `gradient step` a few times.
""")
code(r"""
net = nn.Sequential(nn.Linear(2, 1))
with torch.no_grad():
    net[0].weight[:] = torch.tensor([[0.5, 0.5]]); net[0].bias[:] = 0.0
explore(net, X=[(-1, 1), (1, 1)], y=[1, 0])
""")
md(r"""
## 2. The XOR network on the four cluster centres

Three tanh units, a random start (`seed 0`). Press `10 steps` a few times: when do all four centres end on their own side?
""")
code(r"""
CENTERS = [(-1, 1), (1, 1), (-1, -1), (1, -1)]      # add or move centres here
LABELS  = [1, 0, 0, 1]
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
X, y, groups = clusters(CENTERS, LABELS)
explore(net, X, y, groups)
""")
md(r"""
## 3. The same with clusters of points

50 points around each centre (standard deviation 0.35). Every value is now a mean ± standard deviation over a cluster; the shaded band on each tanh curve shows the cluster's spread.
""")
code(r"""
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
X, y, groups = clusters(CENTERS, LABELS, noise=0.35)
explore(net, X, y, groups)
""")
md(r"""
## 4. ReLU units

Four ReLU units. A unit whose $z$ is negative for a centre passes 0: its outgoing weight has no effect on that centre's loss.
""")
code(r"""
torch.manual_seed(1)
net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
X, y, groups = clusters(CENTERS, LABELS)
explore(net, X, y, groups)
""")
md(r"""
## 5. Two hidden layers

Write any network of this kind as in PyTorch: `nn.Linear` layers with `nn.Tanh`, `nn.ReLU` or `nn.Sigmoid` between them, ending in `nn.Linear(..., 1)`.
""")
code(r"""
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 1))
explore(net, X, y, groups)
""")
md(r"""
## 6. Your own points

Six points on a line: class 1 in the middle. No line separates them; with two tanh units the network can.
""")
code(r"""
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1))
explore(net, X=[(-2, 0), (-1, 0), (0, 0), (0.5, 0), (1, 0), (2, 0)], y=[0, 0, 1, 1, 0, 0])
""")
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "slopewalk_demo.ipynb")
print("cells", len(cells))
