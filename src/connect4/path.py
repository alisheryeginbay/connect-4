from pathlib import Path


def find_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from this file until a folder containing `marker` is found."""
    for folder in Path(__file__).resolve().parents:
        if (folder / marker).exists():
            return folder
    raise FileNotFoundError(f"no {marker} above {__file__}")


ROOT_DIR = find_root()
