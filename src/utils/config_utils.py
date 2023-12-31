from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the configuration file.

    Parameters:
    - config_path (str): The path to the configuration file.

    Returns:
    - dict: The configuration settings.
    """
    with open(config_path, "r") as config_file:
        config: Dict[str, Any] = yaml.safe_load(config_file)
    return config
