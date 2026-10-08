"""Session.record: a GIF of a run, the same weights and history as step(), clear errors for other formats."""

import matplotlib.pyplot as plt
import pytest
from helpers import make
from PIL import Image

from slopewalk import Session, explore, xor_centres

XOR = xor_centres()


def frames(path):
    with Image.open(path) as im:
        sizes = set()
        for i in range(im.n_frames):
            im.seek(i)
            sizes.add(im.size)
        return im.n_frames, sizes


def test_gif_frames_and_state(tmp_path):
    s = Session(make("tanh3"), *XOR, eta=0.5, views=("line",))
    ref = Session(make("tanh3"), *XOR, eta=0.5)
    path = s.record(tmp_path / "run.gif", steps=5, dpi=30)
    n, sizes = frames(path)
    assert n == 6 and len(sizes) == 1  # the start and five steps, all of the same size
    ref.step(5)
    assert s.weights() == pytest.approx(ref.weights()) and s.history == pytest.approx(ref.history)
    assert len(s.predictions) == 5
    assert plt.get_fignums() == []


def test_every_keeps_the_last_frame(tmp_path):
    s = Session(make("tanh3"), *XOR)
    n, _ = frames(s.record(tmp_path / "run.gif", steps=7, every=3, dpi=30))
    assert n == 1 + 2 + 1  # start, after steps 3 and 6, and after the last step


def test_formats(tmp_path, monkeypatch):
    from matplotlib.animation import FFMpegWriter

    s = Session(make("tanh3"), *XOR)
    with pytest.raises(ValueError, match="record writes .gif"):
        s.record(tmp_path / "run.png", steps=1)
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: False))
    with pytest.raises(ValueError, match="needs ffmpeg.*\\.gif"):
        s.record(tmp_path / "run.mp4", steps=1)


def test_record_updates_the_widgets_once(tmp_path, monkeypatch):
    pytest.importorskip("ipywidgets")
    import IPython.display

    monkeypatch.delenv("NB_STATIC", raising=False)
    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: None)
    s = explore(make("tanh3"), *XOR, eta=0.5)
    s.record(tmp_path / "run.gif", steps=3, dpi=30)
    assert s.controller.values() == pytest.approx(s.weights())
    assert len(s.history) == 4
