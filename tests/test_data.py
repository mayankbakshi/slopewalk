"""data.py: clusters, the built-in examples, and the tensors and groups of an explore call."""

import numpy as np
import pytest
import torch

from slopewalk import clusters, two_points, xor_centres
from slopewalk.data import XOR_CENTRES, XOR_LABELS, prepare


def test_xor_centres_in_order():
    X, y, groups = xor_centres()
    assert (
        X.tolist() == [[-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]] == [list(map(float, c)) for c in XOR_CENTRES]
    )
    assert y.tolist() == [1.0, 0.0, 0.0, 1.0] == [float(v) for v in XOR_LABELS]
    assert groups.tolist() == [0, 1, 2, 3]
    Xn, yn, gn = xor_centres(noise=0.35, n_per=100, seed=0)
    Xc, yc, gc = clusters(XOR_CENTRES, XOR_LABELS, noise=0.35, n_per=100, seed=0)
    assert torch.equal(Xn, Xc) and torch.equal(yn, yc) and torch.equal(gn, gc)


def test_two_points():
    X, y, groups = two_points()
    assert X.tolist() == [[-1.0, 1.0], [1.0, 1.0]] and y.tolist() == [1.0, 0.0] and groups.tolist() == [0, 1]


def test_clusters_shapes_and_dtypes():
    X, y, groups = clusters([(0, 0, 0), (1, 1, 1)], [0, 1], noise=0.1, n_per=5, seed=2)
    assert X.shape == (10, 3) and y.shape == (10,) and groups.shape == (10,)
    assert X.dtype == torch.float32 and y.dtype == torch.float32 and groups.dtype == torch.int64
    assert torch.allclose(X[:5].mean(0), torch.zeros(3), atol=0.2) and torch.allclose(
        X[5:].mean(0), torch.ones(3), atol=0.2
    )


def test_prepare_converts_lists_and_integer_labels():
    D = prepare([(-1, 1), (1, 1)], [1, 0])
    assert D.X.dtype == torch.float32 and D.y.dtype == torch.float32 and D.groups.dtype == torch.int64
    assert (D.n, D.d, D.G) == (2, 2, 2) and D.gids == [0, 1]
    D1 = prepare([1.0, 2.0], [1])  # one point given as a flat list
    assert D1.X.shape == (1, 2) and D1.n == 1


def test_prepare_default_groups():
    assert prepare(torch.zeros(8, 2), torch.zeros(8)).groups.tolist() == list(range(8))  # up to 8 points: one each
    y = torch.tensor([0.0, 1.0] * 5)
    assert prepare(torch.zeros(10, 2), y).groups.tolist() == y.long().tolist()  # more: one group per class
    D = prepare(torch.zeros(4, 2), torch.zeros(4), groups=[3, 3, 7, 7])
    assert D.gids == [3, 7] and D.G == 2 and D.gsel[0].tolist() == [True, True, False, False]


def test_stats_mean_and_population_sd():
    D = prepare(torch.zeros(4, 2), torch.zeros(4), groups=[0, 0, 0, 1])
    arr = np.array([1.0, 2.0, 6.0, 5.0], dtype=np.float32)
    m, sd = D.stats(arr, 0)
    assert m == pytest.approx(3.0) and sd == pytest.approx(np.std([1.0, 2.0, 6.0]))  # ddof 0
    m1, sd1 = D.stats(arr, 1)
    assert m1 == 5.0 and sd1 == 0.0
