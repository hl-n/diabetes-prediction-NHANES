import os
from io import StringIO
from typing import Optional, Union

import pandas as pd
import requests

from .data_ingestion import infer_separator, load_dataset


def fetch_metadata(
    file_path: str, url: Optional[str] = None
) -> Union[pd.DataFrame, None]:
    """
    Fetch metadata from the specified URL or
    load it from a file path and return it as a DataFrame.

    Parameters:
    - file_path (str): The path to the metadata file.
    - url (str, optional): The URL of the metadata.

    Returns:
    - pd.DataFrame or None: Metadata as a DataFrame,
      or None if retrieval fails.
    """
    try:
        # Check if the file exists at the specified path
        if os.path.isfile(file_path):
            # Read the metadata from the file path into a Pandas DataFrame
            df = load_dataset(file_path=file_path, index_col="Name")
        else:
            # Send an HTTP GET request to the URL
            response = requests.get(url)
            # Check if the request was successful (status code 200)
            response.raise_for_status()
            # Use StringIO to create a file-like object from the HTML content
            html_content = StringIO(response.text)
            # Read HTML tables from the file-like object
            tables = pd.read_html(html_content, header=0, index_col=0)

            if not tables:
                print("No tables found in the HTML content.")
                return None

            # Assuming the first table contains the metadata
            df = tables[0]

            # Create the directory for the file path
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Save the metadata to the specified file path
            df.to_csv(file_path, sep=infer_separator(file_path), index=True)

        return df

    except Exception as e:
        print(f"Failed to retrieve data. Error: {str(e)}")
        return None


def infer_missing_metadata(
    raw_df: pd.DataFrame, metadata_df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """
    Infer missing metadata information
    based on the raw dataset and provided configuration.

    Parameters:
    - raw_df (pd.DataFrame): The raw dataset.
    - metadata_df (pd.DataFrame): The metadata DataFrame with missing values.
    - config (dict): Configuration settings.
        - max_num_categories (int):
          The maximum number of categories a categorical variable can have.

    Returns:
    - pd.DataFrame: Updated metadata DataFrame.
    """
    for index, row in metadata_df.iterrows():
        if pd.isna(row["Labels"]):
            metadata_df.loc[index, "Labels"] = index.title()
        if pd.isna(row["Levels"]):
            num_categories = raw_df[index].nunique()
            if (raw_df[index].dtype == "object") or (
                num_categories <= config.get("max_num_categories")
            ):
                metadata_df.loc[index, "Levels"] = num_categories
        if pd.isna(row["Class"]):
            if raw_df[index].dtype == "object":
                metadata_df.loc[index, "Class"] = "categorical"
            elif raw_df[index].dtype == "int64":
                metadata_df.loc[index, "Class"] = "integer"
            elif raw_df[index].dtype == "float64":
                metadata_df.loc[index, "Class"] = "numeric"
    return metadata_df
