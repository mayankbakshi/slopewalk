"""slopewalk: gradient descent by hand on small PyTorch networks.

import torch.nn as nn
from slopewalk import explore, clusters

net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
X, y, groups = clusters([(-1, 1), (1, 1), (-1, -1), (1, -1)], [1, 0, 0, 1])
explore(net, X, y, groups)
"""

from .data import clusters, two_points, xor_centres
from .model import weight_table
from .session import Session, explore

__version__ = "0.1.0.dev0"
__all__ = ["explore", "clusters", "xor_centres", "two_points", "weight_table", "Session", "__version__"]
