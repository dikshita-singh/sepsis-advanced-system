import yaml
import os

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Reads the global YAML configuration file and returns a dictionary of settings.
    """
    # If running from inside notebooks or src, adjust path to find config.yaml at root
    if not os.path.exists(config_path):
        config_path = os.path.join("..", "config.yaml")
        
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    return config