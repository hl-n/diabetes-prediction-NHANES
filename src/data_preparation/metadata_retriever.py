from io import StringIO
from typing import Union

import pandas as pd
import requests


def fetch_metadata(url: str) -> Union[pd.DataFrame, None]:
    """
    Fetch metadata from the specified URL and return it as a DataFrame.

    Parameters:
    - url (str): URL of the metadata.

    Returns:
    - pd.DataFrame or None: Metadata as a DataFrame,
      or None if retrieval fails.
    """

    try:
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
        return tables[0]

    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve data from {url}. Error: {e}")
        return None
