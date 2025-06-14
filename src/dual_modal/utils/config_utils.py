import yaml
from pathlib import Path

def load_config(config_path: Path = Path("config.yaml")) -> dict:
    """
    Loads the configuration from a YAML file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config 