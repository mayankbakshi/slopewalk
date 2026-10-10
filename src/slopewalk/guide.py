"""slopewalk.help(): the functions of the package with their arguments, as text in the notebook."""

import inspect
import textwrap

from .data import clusters, two_points, xor_centres
from .model import weight_table
from .session import Session, explore

FUNCTIONS = (explore, clusters, xor_centres, two_points, weight_table)
OPTIONS = (  # the arguments of explore, one line each
    ("net", "nn.Sequential of nn.Linear layers with nn.Tanh, nn.ReLU, nn.Sigmoid or nothing between them"),
    (
        "X, y",
        "the inputs (n x d), and the labels (0 or 1, or 0 to K-1) or the targets (task='regression': n numbers, "
        "or n x K for a network with K outputs)",
    ),
    ("groups", "a number per point, one panel per group; default: one per point up to 8 points, else per class"),
    ("eta", "the step size of a gradient step"),
    ("title", "a title above the picture"),
    ("task", "'binary', 'multiclass' or 'regression'; default: from the network's outputs and y"),
    ("params", "the weights that move: ['w10', 'w11'], 'output' or 'layer:k'; the others stay fixed"),
    ("views", "extra panels: 'backprop', 'cases', 'slice', 'line', 'hidden', 'momentum'"),
    ("case_slopes", "the weight of the 'cases' view, for example 'w10'"),
    ("slice_weights", "one or two weights for the 'slice' view"),
    ("beta, lookahead", "momentum, and the number of steps the 'momentum' view looks ahead"),
    ("log_scale", "R on a log axis"),
    ("layout, width", "'auto', 'wide' or 'standard'; the width of the figure in inches"),
    ("backprop_step", "the stage of the backward pass drawn (a step above 0 turns the view on)"),
    ("backprop_case", "the group whose case the Worked example follows ('A' or 0 is the first)"),
    ("theme", "'auto' (the editor's theme where the kernel can read it), 'light' or 'dark'"),
)
SESSION = (
    "step",
    "step_back",
    "reset",
    "weights",
    "risk",
    "slopes",
    "predict",
    "lookahead",
    "case_slopes",
    "save",
    "record",
    "record_backprop",
)


def first_line(obj):
    """The first sentence of an object's docstring."""
    doc = inspect.getdoc(obj) or ""
    return doc.split("\n\n")[0].replace("\n", " ").split(". ")[0].rstrip(".") + "."


def signature(name, obj, width=100):
    """name(arguments), wrapped under the opening parenthesis; a method without self."""
    sig = inspect.signature(obj)
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        sig = sig.replace(parameters=params[1:])
    text = f"{name}{sig}"
    return textwrap.fill(text, width=width, subsequent_indent=" " * (len(name) + 1), break_long_words=False)


def help(about=None):  # kept out of __all__, so that a star import leaves Python's own help alone
    """Prints the functions of slopewalk with their arguments, the options of explore and the methods of the session
    it returns. help("explore"), help("Session.step") or help(clusters) prints the full description of one
    of them."""
    if about is not None:
        print(describe(about))
        return
    lines = ["slopewalk: gradient descent by hand on small PyTorch networks.", ""]
    for f in FUNCTIONS:
        lines += [
            signature(f.__name__, f),
            textwrap.fill(first_line(f), 100, initial_indent="    ", subsequent_indent="    "),
        ]
    lines += ["", "Arguments of explore:"]
    lines += [
        textwrap.fill(text, 100, initial_indent=f"    {name:<17}", subsequent_indent=" " * 21) for name, text in OPTIONS
    ]
    lines += ["", "explore returns a Session; its methods:"]
    lines += [textwrap.indent(signature(m, getattr(Session, m), width=96), "    ") for m in SESSION]
    lines += ["    and history (R after each step), predictions, path (the weights after each step), w0 (the start)."]
    lines += ["", 'slopewalk.help("explore") or slopewalk.help("Session.step") prints the full description of one.']
    print("\n".join(lines))


def describe(about):
    """The signature and the whole docstring of a function or method, given by name or as the object."""
    if isinstance(about, str):
        names = {f.__name__: f for f in FUNCTIONS} | {f"Session.{m}": getattr(Session, m) for m in SESSION}
        names["Session"] = Session
        if about not in names:
            raise ValueError(f"slopewalk.help: {about!r} is not one of {', '.join(names)}.")
        name, obj = about, names[about]
    else:
        name, obj = getattr(about, "__qualname__", repr(about)), about
    return signature(name, obj) + "\n\n" + textwrap.indent(inspect.getdoc(obj) or "", "    ")
