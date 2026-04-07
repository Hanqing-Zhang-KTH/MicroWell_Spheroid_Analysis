"""
Research group:
Björn Önfelt Group
Department of Applied Physics, Division of Biophysics
Royal Institute of Technology

Coding author:
Hanqing Zhang, Researcher, Royal Institute of Technology, hanzha@kth.se
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tifffile
import psutil
import matplotlib.pyplot as plt

from .io_utils import ensure_dir


def _estimate_stack_bytes(stack: np.ndarray) -> int:
    return int(stack.size * stack.dtype.itemsize)


def _check_memory_for_stacks(stacks: Dict[str, np.ndarray], safety_fraction: float = 0.5) -> None:
    """Raise RuntimeError if estimated memory for stacks exceeds a fraction of available RAM."""
    total_bytes = sum(_estimate_stack_bytes(arr) for arr in stacks.values())
    available = psutil.virtual_memory().available
    if total_bytes > safety_fraction * available:
        raise RuntimeError(
            f"Estimated memory for stacks ({total_bytes / 1e9:.2f} GB) exceeds "
            f"{safety_fraction:.0%} of available RAM ({available / 1e9:.2f} GB)."
        )


def fixed_global_thresholding(
    nucleus_stack: np.ndarray,
    tumor_stack: np.ndarray,
    fibro_stack: np.ndarray,
    thresholds: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """
    Apply fixed global thresholds to three intensity stacks.

    Returns a dict with boolean masks for keys: 'nucleus', 'tumor', 'fibroblast'.
    """
    stacks = {
        "nucleus": nucleus_stack,
        "tumor": tumor_stack,
        "fibroblast": fibro_stack,
    }
    _check_memory_for_stacks(stacks)

    nuc_thr = float(thresholds["nucleus_draq5"])
    tumor_thr = float(thresholds["tumor_rfp"])
    fibro_thr = float(thresholds["fibroblast_gfp"])

    masks = {
        "nucleus": (nucleus_stack >= nuc_thr),
        "tumor": (tumor_stack >= tumor_thr),
        "fibroblast": (fibro_stack >= fibro_thr),
    }
    return masks


def save_binary_stack(mask: np.ndarray, out_path: Path) -> None:
    """Save a 3D boolean mask as TIFF with values 0 / 255."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    mask_uint8 = (mask.astype(np.uint8)) * 255
    with tifffile.TiffWriter(str(out_path)) as writer:
        for z in range(mask_uint8.shape[0]):
            writer.write(mask_uint8[z])


def _middle_slice(stack: np.ndarray) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {stack.shape}")
    z = stack.shape[0] // 2
    return stack[z]


def save_threshold_vs_manual_figures(
    threshold_mask: np.ndarray,
    manual_mask: np.ndarray,
    out_base: Path,
    title_prefix: str,
) -> None:
    """
    Save side-by-side and color-overlap visualizations of one middle slice.

    threshold_mask and manual_mask are boolean 3D stacks with the same shape.
    """
    out_base = Path(out_base)
    ensure_dir(out_base.parent)

    thr_mid = _middle_slice(threshold_mask.astype(bool))
    man_mid = _middle_slice(manual_mask.astype(bool))

    # 1) Side-by-side binary masks
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(man_mid, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"{title_prefix} – manual mask")
    axes[0].axis("off")
    axes[1].imshow(thr_mid, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"{title_prefix} – threshold mask")
    axes[1].axis("off")
    plt.tight_layout()
    side_path = out_base.parent / f"{out_base.name}_side_by_side.png"
    fig.savefig(str(side_path), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Overlap visualization with distinct colors:
    # manual mask = bright blue, threshold mask = purple, overlap = red
    h, w = thr_mid.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    # manual: blue
    overlay[man_mid, 2] = 1.0
    # threshold: purple (red + blue)
    overlay[thr_mid, 0] = np.maximum(overlay[thr_mid, 0], 0.6)
    overlay[thr_mid, 2] = np.maximum(overlay[thr_mid, 2], 0.6)
    # overlap: red dominates
    overlap_mask = man_mid & thr_mid
    overlay[overlap_mask, :] = 0.0
    overlay[overlap_mask, 0] = 1.0
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(overlay)
    ax.set_title(f"{title_prefix} – overlap (R: manual, G: threshold)")
    ax.axis("off")
    plt.tight_layout()
    overlap_path = out_base.parent / f"{out_base.name}_overlap.png"
    fig.savefig(str(overlap_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mask_overlap_figure(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """
    Save an overlap figure between two 3D binary masks using a middle slice.
    First mask is shown in bright blue, second in purple, and overlap in red.
    """
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    mid_a = _middle_slice(mask_a)
    mid_b = _middle_slice(mask_b)

    h, w = mid_a.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)

    # A: blue
    overlay[mid_a, 2] = 1.0
    # B: purple
    overlay[mid_b, 0] = np.maximum(overlay[mid_b, 0], 0.6)
    overlay[mid_b, 2] = np.maximum(overlay[mid_b, 2], 0.6)
    # Overlap: red
    overlap_mask = mid_a & mid_b
    overlay[overlap_mask, :] = 0.0
    overlay[overlap_mask, 0] = 1.0

    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


