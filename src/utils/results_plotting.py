from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from .file_utils import rename_file


def add_annotations_to_bars(ax: Axes, x_shift: int) -> None:
    """
    Add annotations to bars in a horizontal bar plot.

    Parameters:
    - ax (Axes): The axes on which the annotations will be added.
    - x_shift (int):
      The number of points to shift the annotations along the x-axis.

    Returns:
    - None
    """
    for index, p in enumerate(ax.patches):
        percentage = p.get_width()
        if percentage > 0:
            ax.annotate(
                f"{percentage:.1f}%",
                (p.get_width() / 2, p.get_y() + p.get_height() / 2.0),
                ha="center",
                va="center",
                xytext=(x_shift, 0),
                textcoords="offset points",
            )


def save_plot(folder_path: str, title: str) -> None:
    """
    Save a plot as a PNG file.

    Parameters:
    - folder_path (str): The path to the folder where the plot will be saved.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    file_name = f"{rename_file(title)}.png"
    plt.savefig(
        fname=f"{folder_path}{file_name}", bbox_inches="tight"
    )  # Save the plot as a PNG file


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    num_dp: int,
    figsize: Optional[Tuple[float, float]] = None,
    save: bool = False,
    folder_path: Optional[str] = None,
) -> None:
    """
    Plot a heatmap of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be visualized.
    - title (str): The title of the heatmap.
    - num_dp (int):
      The number of decimal places to display in the annotations.
    - figsize (Tuple[float, float], optional):
      A tuple of the width and height of the figure in inches.
      Default is None.
    - folder_path (str, optional):
      The path to the folder where the plot will be saved.
        Default is None, i.e. the figure is not saved.

    Returns:
    - None
    """
    if figsize:
        plt.figure(figsize=figsize)
    sns.heatmap(
        df, annot=True, fmt=f".{num_dp}f", cmap="coolwarm", mask=np.triu(df)
    )
    plt.title(title)
    if folder_path:
        save_plot(folder_path=folder_path, title=title)
    plt.show()
