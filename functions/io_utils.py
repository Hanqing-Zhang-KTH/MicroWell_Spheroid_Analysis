"""
Research group:
Björn Önfelt Group
Department of Applied Physics, Division of Biophysics
Royal Institute of Technology

Coding author:
Hanqing Zhang, Researcher, Royal Institute of Technology, hanzha@kth.se, hanzha@kth.se
"""

import json
import os
from pathlib import Path
from typing import Any, Dict


def load_json_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a JSON configuration file with UTF-8 encoding."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory if it does not exist and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def join(*parts: str | os.PathLike) -> Path:
    """Path join helper that always returns a Path."""
    return Path(parts[0]).joinpath(*parts[1:])


