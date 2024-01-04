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


def plot_distribution(
    df: pd.DataFrame, folder_path: Optional[str] = None
) -> None:
    """
    Plot the distribution of numerical features in a DataFrame.

    Parameters:
        - df (pd.DataFrame): The DataFrame to be visualized.
        - folder_path (str, optional):
          The path to the folder where the plot will be saved.
          Default is None.

    Returns:
        - None
    """
    # Assuming numerical_feats contains the numerical column names
    features = df.columns

    # Set the number of rows and columns for subplots
    ncols = 3
    nrows = (len(features) // ncols) + (len(features) % ncols)

    # Create subplots
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, 2.5 * nrows), constrained_layout=True
    )

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate through features and plot the distribution in subplots
    for i, feat in enumerate(features):
        if df[feat].dtype.kind not in "biufc":
            sns.histplot(y=df[feat], kde=True, ax=axes[i])
        else:
            sns.histplot(df[feat], kde=True, ax=axes[i])

        axes[i].set_title(feat)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

        # Check if the variable is boolean and set x-ticks accordingly
        if df[feat].dtype == "bool":
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(["False", "True"])

    # Remove empty subplots
    for j in range(len(features), nrows * ncols):
        fig.delaxes(axes[j])

    title = "Distribution of Features"
    plt.suptitle(title, x=0.62)

    if folder_path:
        # Save the plot as a PNG file
        save_plot(folder_path=folder_path, title=title)

    # Show the plot
    plt.show()


def plot_ratio_target_in_categorical_feature(
    df: pd.DataFrame, feature: str, target: str, ax: Axes
) -> None:
    """
    Plot the prevalence of a target variable
    across different categories of a categorical feature.

    Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        - feature (str): The categorical feature to be visualized.
        - target (str): The target variable.
        - ax (matplotlib.axes.Axes):
          The axes on which the annotations will be added.

    Returns:
        - None
    """
    # Calculate the proportion of target group
    # for each category in column_name
    proportion_target_by_feature = df.groupby(feature)[target].mean() * 100
    sns.barplot(
        ax=ax,
        y=proportion_target_by_feature.index.astype(str),
        x=proportion_target_by_feature.values,
        color="skyblue",
    )

    add_annotations_to_bars(ax=ax, x_shift=8)

    ax.set_title(feature)
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_ratio_target_in_categorical_features(
    df: pd.DataFrame, config: dict, folder_path: Optional[str] = None
) -> None:
    """
    Plot the percentage of the target variable
    by different categorical features.

    Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        - config (dict): Configuration settings.
        - folder_path (str, optional):
          The path to the folder where the plot will be saved.
          Default is None.

    Returns:
        - None
    """
    target = config.get("target")
    categorical_features = df.columns[
        df.nunique() <= config.get("max_num_categories")
    ]
    categorical_features = categorical_features.drop(target)
    non_binary_features = categorical_features[
        df[categorical_features].nunique() > 2
    ]
    categorical_features = categorical_features.drop(
        non_binary_features
    ).to_list()
    categorical_features = categorical_features + non_binary_features.to_list()
    num_features = len(categorical_features)
    nrows = num_features // 2 + num_features % 2
    ncols = 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10, 8),
        constrained_layout=True,
        gridspec_kw={
            "height_ratios": [1] * (nrows - 2) + [11 / 2, 5 / 2],
        },
    )
    for i, feature in enumerate(categorical_features, start=0):
        ax = axes.flatten()[i]
        plot_ratio_target_in_categorical_feature(
            df=df, feature=feature, target=target, ax=ax
        )
    # Remove empty subplots
    for j in range(num_features, nrows * ncols):
        fig.delaxes(axes.flatten()[j])

    title = f"Percentage of {target} by Categorical Features"

    plt.suptitle(title, x=0.62)
    if folder_path:
        save_plot(folder_path=folder_path, title=title)
    plt.show()
