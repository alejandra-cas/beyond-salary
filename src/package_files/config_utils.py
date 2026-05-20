from pathlib import Path

import yaml


def get_repo_root():
    """Return the repository root based on this module's location."""
    return Path(__file__).resolve().parents[2]


def get_processed_dir():
    """Load processed_dir from config.yaml, with a repo-local fallback."""
    repo_root = get_repo_root()
    config_path = repo_root / "config.yaml"

    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["data"]
        processed_dir = Path(cfg["processed_dir"])
        if not processed_dir.is_absolute():
            processed_dir = repo_root / processed_dir
        return processed_dir

    print(
        "Warning: config.yaml not found, using default processed paths. "
        "Copy config.example.yaml to config.yaml to configure."
    )
    return repo_root / "data" / "processed"
