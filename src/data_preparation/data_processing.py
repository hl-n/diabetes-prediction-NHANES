from typing import Dict, Optional

import pandas as pd


def convert_range_to_min_max(
    df: pd.DataFrame,
    config: Dict[str, Optional[str]],
    drop_column: bool = True,
) -> pd.DataFrame:
    """
    Converts a range column into two separate columns:
    'Minimum' and 'Maximum'.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the range column.
    - config (Dict[str, Optional[str]]): Configuration parameters.
        - range_column (str): The name of the range column.
    - drop_column (bool): Defaults to True,
      in which case the original column is removed from the output.

    Returns:
    - pd.DataFrame: The DataFrame with two additional columns:
      'Minimum' and 'Maximum'.
    """

    column = config.get("range_column")

    # Define regular expression patterns
    pattern_brackets_lower = r"(?:\[(\d+),|>= (\d+)|> (\d+))"
    pattern_brackets_upper = r"(?:,(\d+)\)|<= (\d+)|< (\d+))"

    # Extract lower and upper limits using separate patterns
    df[f"Minimum {column}"] = (
        df[column].str.extract(pattern_brackets_lower).fillna("").sum(axis=1)
    )
    df[f"Maximum {column}"] = (
        df[column].str.extract(pattern_brackets_upper).fillna("").sum(axis=1)
    )

    # Convert the new columns to numeric values
    df[f"Minimum {column}"] = pd.to_numeric(df[f"Minimum {column}"])
    df[f"Maximum {column}"] = pd.to_numeric(df[f"Maximum {column}"])

    if drop_column:
        df.drop(columns=[column], inplace=True)

    return df


def convert_column_to_boolean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Convert a categorical column to binary indicator columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Name of the column to be converted.

    Returns:
    - pd.DataFrame: DataFrame with binary indicator columns added.
    """
    value = df[column].iloc[0]
    df = df.copy()
    df[f"Is {value.title()}"] = df[column].eq(value)
    return df.drop(columns=[column])


def convert_columns_to_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all suitable columns in the DataFrame to binary indicator columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with binary indicator columns added.
    """
    for column in df.columns:
        if (
            (df[column].notna().all())
            and (df[column].nunique() == 2)
            and (df[column].dtype == "object")
        ):
            df = convert_column_to_boolean(df, column)
    return df


def one_hot_encode_column(
    df: pd.DataFrame, column: str, drop_column: bool = True
) -> pd.DataFrame:
    """
    One-hot encode a categorical column and
    add the resulting columns to the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Name of the column to be one-hot encoded.
    - drop_column (bool): Defaults to True,
    in which case the original categorical column is removed from the output.

    Returns:
    - pd.DataFrame: DataFrame with one-hot encoded columns added.
    """

    if drop_column:
        prefix = column
    else:
        prefix = None

    df = pd.concat([df, pd.get_dummies(df[column], prefix=prefix)], axis=1)

    if drop_column:
        df.drop(columns=[column], inplace=True)

    return df


def one_hot_encode_columns(
    df: pd.DataFrame, drop_columns: bool = True
) -> pd.DataFrame:
    """
    One-hot encode all suitable columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - drop_columns (bool): Defaults to True,
    in which case the original categorical columns are
    removed from the output.

    Returns:
    - pd.DataFrame: DataFrame with one-hot encoded columns added.
    """
    for column in df.columns:
        if df[column].dtype == "object":
            df = one_hot_encode_column(df, column, drop_columns)
    return df


def create_target_variable(
    df: pd.DataFrame, config: Dict[str, Optional[str]]
) -> pd.DataFrame:
    """
    Create a binary target column based on a specified threshold.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - config (Dict[str, Optional[str]]): Configuration parameters.
        - feature (str): The feature column for thresholding.
        - threshold (float):
          The threshold for creating the binary target column.
        - target_column (str): Name of the target column.

    Returns:
    - pd.DataFrame: DataFrame with the binary target column added.
    """
    # Remove rows without data for feature
    df = df[df[config.get("feature")].notna()]
    df[config.get("target")] = df[config.get("feature")] >= config.get(
        "threshold"
    )
    return df


def process_data(
    df: pd.DataFrame, config: Dict[str, Optional[str]]
) -> pd.DataFrame:
    """
    Perform various data processing operations on the DataFrame
    based on configuration parameters.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - config (Dict[str, Optional[str]]): Configuration parameters.
        - range_column (str): The name of the range column.
        - feature (str): The feature column for thresholding.
        - threshold (float):
          The threshold for creating the binary target column.
        - target_column (str): Name of the target column.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    df = convert_range_to_min_max(df, config)
    df = convert_columns_to_boolean(df)
    df = one_hot_encode_columns(df)
    df = create_target_variable(df, config)
    return df
