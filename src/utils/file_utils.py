import os
from typing import Dict, Optional


def rename_file(file_name: str) -> str:
    """
    Rename a file by replacing '/' with ' or ', spaces with underscores, and
    deleting other special characters.

    Parameters:
    - file_name (str): The name of the file to be renamed.

    Returns:
    - str: The modified file name.
    """
    translation_table = str.maketrans(
        {
            "/": " or ",
            " ": "_",
            "$": "",
            "\\": "",
        }
    )
    return file_name.translate(translation_table)


def create_results_directory(
    config: Dict[str, Optional[str]], stage: str, folder_name: str = None
) -> None:
    """
    Create a directory for storing results based on configuration parameters.

    Parameters:
    - config (Dict[str, Optional[str]]): Configuration parameters.
        - imputation_methods (Dict[str, str]):
          Dictionary of imputation methods.
        - imputation_method (str): Chosen imputation method.
        - results_folder_path (str): Base folder path for storing results.
    - stage (str): The stage for which the directory is being created
      (e.g., 'eda' or 'modeling').

    Returns:
    - None
    """
    if folder_name is None:
        imputation_methods = config.get("imputation_methods")
        imputation_method = config.get("imputation_method")
        method_name = imputation_methods.get(imputation_method)
        folder_name = rename_file(method_name)
    results_folder_path = config.get("results_folder_path")
    folder_path = f"{results_folder_path}{stage}/{folder_name}/"
    os.makedirs(os.path.dirname(folder_path), exist_ok=True)
    return folder_path
