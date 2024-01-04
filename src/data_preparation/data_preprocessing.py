from typing import Any, Dict, Optional

import pandas as pd

from .metadata_retriever import fetch_metadata, infer_missing_metadata


def rename_columns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Rename DataFrame columns based on metadata information.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.
        config (Dict): Configuration parameters.
            - "metadata_url" (str): URL for fetching metadata.
            - "metadata_path" (str): File path for metadata.

    Returns:
        pd.DataFrame: DataFrame with columns renamed
        based on metadata labels and units.
    """
    metadata_df = fetch_metadata(
        url=config.get("metadata_url"), file_path=config.get("metadata_path")
    )
    # Clean metadata
    metadata_df = infer_missing_metadata(df, metadata_df, config)
    # Add escape character "\\" before the percentage symbol "%"
    # to tell LaTeX to treat the "%" as a regular character.
    metadata_df["Units"] = metadata_df["Units"].replace("%", "\\%")
    # Rename columns by their labels and units
    df.columns = [
        (
            f'{metadata_df.loc[col, "Labels"]} '
            f'(${metadata_df.loc[col, "Units"]}$)'
        )
        if pd.notnull(metadata_df.loc[col, "Units"])
        else f'{metadata_df.loc[col, "Labels"]}'
        for col in df.columns
    ]
    return df


def index_by_identifier(
    df: pd.DataFrame, config: Dict[str, str]
) -> pd.DataFrame:
    """
    Set DataFrame index using a specified unique identifier column.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.
        config (Dict): Configuration parameters.
            - "unique_identifier" (str): Name of the column to be used as the
              unique identifier for setting the DataFrame index.

    Returns:
        pd.DataFrame: DataFrame with the index set to the
        specified unique identifier column.
    """
    df.index = df[config.get("unique_identifier")]
    return df


def preprocess_data(
    df: pd.DataFrame, config: Dict[str, Optional[str]]
) -> pd.DataFrame:
    """
    Perform various data preprocessing operations on the DataFrame
    based on configuration parameters.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - config (Dict[str, Optional[str]]): Configuration parameters.
        - "metadata_url" (str): URL for fetching metadata.
        - "metadata_path" (str): File path for metadata.
        - "unique_identifier" (str): Name of the column to be used as the
          unique identifier for setting the DataFrame index.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    df = rename_columns(df, config)
    df = index_by_identifier(df, config)

    return df
