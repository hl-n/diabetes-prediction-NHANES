import os
from typing import Optional

import pandas as pd


def load_dataset(
    file_path: str, url: Optional[str] = None, index_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Load the dataset from file path or URL into a Pandas DataFrame.

    Parameters:
    - file_path (str): The path to the dataset file.
    - url (str, optional): The URL of the dataset file.

    Returns:
    - pd.DataFrame: The loaded dataset.

    Raises:
    - ValueError: If the file path does not exist and the URL is unreachable
      or if there is an issue with loading the dataset.
    """
    try:
        # Check if the file exists at the specified path
        if os.path.isfile(file_path):
            # Read the dataset from the file path into a Pandas DataFrame
            df = pd.read_csv(
                file_path, sep=infer_separator(file_path), index_col=index_col
            )
        elif url is not None:
            # Read the dataset from the URL into a Pandas DataFrame
            df = pd.read_csv(url, sep=infer_separator(url))
            # Create the directory for the file path
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Save the dataset to the specified file path
            df.to_csv(file_path, sep=infer_separator(file_path), index=False)
        else:
            raise ValueError(
                "Please provide either a valid URL or an existing file path."
            )

        return df
    except Exception as e:
        raise ValueError(f"Failed to load dataset. Error: {str(e)}")


def infer_separator(file_path: str) -> str:
    """
    Infer the separator for a dataset file
    based on the extension in its file path.

    Parameters:
    - file_path (str): The path to the dataset file.

    Returns:
    - str: The separator used in the dataset file

    Raises:
    - ValueError: If the file type is not supported
      or if there's an issue inferring the delimiter.
    """
    try:
        # Extract the file type (extension) from the file name
        _, file_extension = os.path.splitext(file_path)

        # Determine the delimiter based on the file type
        separator_mapping = {".csv": ",", ".tsv": "\t", ".txt": "\t"}

        # Use the determined delimiter
        # or raise an error if the file type is not supported
        separator = separator_mapping.get(file_extension)
        if separator is None:
            raise ValueError(
                f"Unsupported file type: {file_extension}. "
                f"Supported types: {', '.join(separator_mapping.keys())}"
            )
        return separator
    except Exception as e:
        raise ValueError(
            f"Failed to infer separator for {file_path}. " f"Error: {str(e)}"
        )
