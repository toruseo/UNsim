"""
General utilities for UNsim.
"""

import warnings
import functools
import traceback
import sys

import numpy as np
import matplotlib.pyplot as plt



def display_image_in_notebook(image_path):
    """
    Display an image in Jupyter Notebook.

    Parameters
    ----------
    image_path : str
        The path to the image file to display.
    """
    from IPython.display import display, Image
    with open(image_path, "rb") as f:
        display(Image(data=f.read(), format='png'))
