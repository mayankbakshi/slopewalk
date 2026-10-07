"""Still pictures: with NB_STATIC=1 (HTML copies of notebooks), or when ipywidgets is missing."""

import matplotlib.pyplot as plt

STATIC_MESSAGE = (
    "Still picture of the start. "
    "Run the notebook (Colab, or Jupyter / VS Code with ipywidgets) to get the sliders and buttons."
)
NO_WIDGETS_MESSAGE = (
    "No sliders: the package ipywidgets is missing. "
    "Run  %pip install ipywidgets  in a cell, restart the kernel and run again,\n"
    "or open the notebook in Google Colab, where ipywidgets is installed."
)


def still_picture(session, message):
    """Prints the message and draws the start once. Returns the session."""
    print(message)
    session.draw()
    plt.show()
    return session
