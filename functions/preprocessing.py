"""
Research group:
Björn Önfelt Group
Department of Applied Physics, Division of Biophysics
Royal Institute of Technology

Coding author:
Hanqing Zhang, Researcher, Royal Institute of Technology, hanzha@kth.se
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import gaussian_filter


def apply_optional_gaussian(
    stack: np.ndarray,
    cfg: Dict[str, object],
) -> np.ndarray:
    """Apply Gaussian blur if enabled in config."""
    if not bool(cfg.get("enable_gaussian_blur", False)):
        return stack
    sigma = float(cfg.get("gaussian_sigma", 1.0))
    return gaussian_filter(stack, sigma=sigma)


