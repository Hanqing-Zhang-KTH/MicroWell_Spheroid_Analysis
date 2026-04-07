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

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import feature, morphology, segmentation, color as skcolor

from .io_utils import ensure_dir


def _middle_slice(stack: np.ndarray) -> np.ndarray:
    z = stack.shape[0] // 2
    return stack[z]


def _save_stack_binary(mask: np.ndarray, out_path: Path) -> None:
    mask = (mask.astype(np.uint8)) * 255
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    with tifffile.TiffWriter(str(out_path)) as writer:
        for z in range(mask.shape[0]):
            writer.write(mask[z])


def _instance_projection_to_rgb(labels_3d: np.ndarray, intensity_stack: np.ndarray) -> np.ndarray:
    labels_3d = np.asarray(labels_3d, dtype=np.int32)
    roi = np.asarray(intensity_stack, dtype=np.float64)
    if labels_3d.shape != roi.shape:
        raise ValueError(f"labels_3d and intensity_stack must match shape, got {labels_3d.shape} vs {roi.shape}")
    Z, Y, X = labels_3d.shape
    roi_masked = np.where(labels_3d > 0, roi, -np.inf)
    z_max = np.argmax(roi_masked, axis=0)
    proj_label = labels_3d[z_max, np.arange(Y)[:, None], np.arange(X)[None, :]]
    rgb = skcolor.label2rgb(proj_label, bg_label=0, bg_color=(0, 0, 0))
    return (np.clip(rgb * 255, 0, 255)).astype(np.uint8)


def _mask_cell_boundaries_to_zero(labels: np.ndarray) -> np.ndarray:
    """Set cell boundary voxels to 0 so the output mask has clear gaps between cells."""
    if labels.max() <= 0:
        return (labels > 0).astype(bool)
    max_l = ndi.maximum_filter(labels, size=3)
    min_l = ndi.minimum_filter(labels, size=3)
    is_interior = (labels > 0) & (max_l == min_l)
    return is_interior


def _refine_with_watershed(
    binary_mask: np.ndarray,
    intensity_stack: np.ndarray,
    cfg: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    3D watershed refinement following Python_Post_Ref/functions/betacell_processing.py
    _refine_segmentation_with_watershed workflow (Averaging3DPatternAnalysis Step 2).

    Steps: 1) closing, 2) Gaussian smooth + threshold → mask_smooth, 3) DoG on intensity,
    4) markers from peak_local_max, 5) watershed, 6) size filter, 7) final smooth,
    8) mask cell boundaries to zero for clear cell separation.

    Returns (final_binary_mask, instance_labels).
    """
    mask = np.asarray(binary_mask > 0, dtype=bool)
    roi = np.asarray(intensity_stack, dtype=np.float32)
    orig_shape = mask.shape

    downscale = float(cfg.get("watershed_downsize_ratio", 1.0))
    if downscale > 1.0:
        scale = 1.0 / downscale
        mask = ndi.zoom(mask.astype(np.float32), scale, order=0) > 0.5
        roi = ndi.zoom(roi, scale, order=1)
        if not np.any(mask):
            mask = np.asarray(binary_mask > 0, dtype=bool)
            roi = np.asarray(intensity_stack, dtype=np.float32)

    # Step 1 – morphological closing (Python_Post_Ref Step 2)
    window_size = max(1, int(cfg.get("window_size", 3)))
    selem = morphology.ball(window_size)
    mask_closed = morphology.closing(mask, selem)

    # Step 2 – Gaussian smooth + threshold → mask_smooth (Python_Post_Ref Step 3)
    smooth_divisor = float(cfg.get("smooth_kernel_divisor", 2.0))
    smooth_thr = float(cfg.get("smooth_threshold1", 0.5))
    smoothed = ndi.gaussian_filter(mask_closed.astype(np.float32), sigma=window_size / smooth_divisor)
    mask_smooth = (smoothed > smooth_thr).astype(bool)
    if not np.any(mask_smooth) and np.any(mask_closed):
        mask_smooth = mask_closed

    # Step 3 – DoG on intensity inside mask_smooth (Python_Post_Ref: cellsize_dog_kernel)
    dog_kernel = cfg.get("cellsize_dog_kernel", [3.0, 8.0])
    if isinstance(dog_kernel, (list, tuple)) and len(dog_kernel) >= 2:
        sigma_small = float(dog_kernel[0])
        sigma_large = float(dog_kernel[1]) + 1.0
    else:
        sigma_small = float(cfg.get("dog_sigma_small", 3.0))
        sigma_large = float(cfg.get("dog_sigma_large", 9.0))
    dog = ndi.gaussian_filter(roi, sigma=sigma_small) - ndi.gaussian_filter(roi, sigma=sigma_large)
    dog = np.maximum(dog, 0.0)
    dog[~mask_smooth] = 0.0

    # Distance map
    distance = ndi.distance_transform_edt(mask_smooth)

    # Step 4 – markers from peak_local_max (Python_Post_Ref: exclude_border=True)
    min_dist = max(1, int(cfg.get("watershed_marker_min_distance_pixels", cfg.get("watershed_marker_min_distance", 5))))
    dog_min = max(0.0, float(cfg.get("watershed_marker_dog_min", cfg.get("watershed_marker_min_dog", 0.0))))
    peaks = feature.peak_local_max(
        dog,
        min_distance=min_dist,
        threshold_abs=dog_min,
        exclude_border=True,
        labels=mask_smooth.astype(np.int32),
    )
    markers = np.zeros(mask_smooth.shape, dtype=np.int32)
    for i, coord in enumerate(peaks):
        markers[tuple(coord)] = i + 1
    if markers.max() == 0:
        markers[mask_smooth] = 1

    # Step 5 – elevation and watershed
    elev = str(cfg.get("watershed_elevation", cfg.get("distance_elevation", "distance"))).strip().lower()
    if elev == "dog":
        elevation = -dog.astype(np.float64)
    else:
        elevation = -distance.astype(np.float64)

    labels = segmentation.watershed(elevation, markers=markers, mask=mask_smooth)

    # Step 6 – remove small regions (Python_Post_Ref: min_cell_volume_pixels)
    min_vol = int(cfg.get("min_cell_volume_voxels", cfg.get("min_cell_volume_pixels", 1000)))
    if min_vol > 0 and labels.max() > 0:
        for lab in range(1, labels.max() + 1):
            if np.sum(labels == lab) < min_vol:
                labels[labels == lab] = 0

    # Step 7 – final smooth + threshold (Python_Post_Ref)
    sigma_final = float(cfg.get("final_smooth_sigma", 0.6))
    thr_final = float(cfg.get("final_smooth_threshold", 0.5))
    binary = (labels > 0).astype(np.float32)
    smooth_final = ndi.gaussian_filter(binary, sigma=sigma_final)
    final_mask = (smooth_final > thr_final).astype(bool)

    # Upsample if we downscaled for watershed
    if downscale > 1.0 and mask.shape != orig_shape:
        zoom_factors = [orig_shape[i] / labels.shape[i] for i in range(3)]
        labels = np.round(ndi.zoom(labels.astype(np.float32), zoom_factors, order=0)).astype(np.int32)
        final_mask = ndi.zoom(final_mask.astype(np.float32), zoom_factors, order=0) > 0.5

    # Step 8 – mask cell boundaries to zero (clear gaps between cells)
    final_mask = _mask_cell_boundaries_to_zero(labels) & final_mask

    return final_mask, labels.astype(np.int32)


def run_post_analysis(
    intensity_stack: np.ndarray,
    binary_mask: np.ndarray,
    cfg: Dict[str, object],
    refinement_dir: Path,
    channel_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run optional mask refinement using 3D watershed and save intermediate
    diagnostics in the 'Refinement' subfolder.

    Returns (refined_binary_mask, instance_labels).
    """
    refinement_dir = ensure_dir(refinement_dir)
    if not bool(cfg.get("enable_watershed_refinement", False)):
        # Still produce an instance map via connected components for consistency
        labels, _ = ndi.label(binary_mask.astype(bool))
        final_mask = _mask_cell_boundaries_to_zero(labels) & binary_mask.astype(bool)
        return final_mask, labels.astype(np.int32)

    final_mask, labels = _refine_with_watershed(binary_mask, intensity_stack, cfg)

    # Save TIFF: cell-separated binary mask (0 or 255)
    _save_stack_binary(final_mask, refinement_dir / f"{channel_name}_refined_mask.tiff")

    # Save TIFF: instance segmentation (grayscale, each instance = integer value)
    inst = labels.astype(np.uint16)
    with tifffile.TiffWriter(str(refinement_dir / f"{channel_name}_instances.tiff")) as writer:
        for z in range(inst.shape[0]):
            writer.write(inst[z])

    # Center-slice data (intensity and mask use channel's own data; nucleus mask used for nucleus cell separation)
    mid_raw = _middle_slice(intensity_stack)
    mid_mask = _middle_slice(final_mask)
    mid_labels = _middle_slice(labels)

    # 1. PNG: center-slice mask (binary)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow((mid_mask.astype(np.uint8) * 255), cmap="gray")
    ax.set_title(f"{channel_name} refined mask (center slice)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(refinement_dir / f"{channel_name}_center_slice_mask.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. PNG: center-slice instance segmentation (color-coded, each object its own color)
    mid_label_rgb = skcolor.label2rgb(mid_labels, bg_label=0, bg_color=(0, 0, 0))
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow((np.clip(mid_label_rgb * 255, 0, 255)).astype(np.uint8))
    ax.set_title(f"{channel_name} instance segmentation (center slice)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(refinement_dir / f"{channel_name}_center_slice_instances.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. PNG: overlap of instance-colored semi-transparent mask on intensity
    int_norm = np.clip((mid_raw - mid_raw.min()) / (mid_raw.max() - mid_raw.min() + 1e-8), 0, 1)
    int_rgb = np.stack([int_norm] * 3, axis=-1)
    inst_rgb = skcolor.label2rgb(mid_labels, bg_label=0, bg_color=(0, 0, 0))
    alpha = 0.5  # semi-transparent instance overlay
    overlay = np.where(
        mid_labels[..., None] > 0,
        alpha * int_rgb + (1 - alpha) * inst_rgb,
        int_rgb,
    )
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(np.clip(overlay, 0, 1))
    ax.set_title(f"{channel_name} instances on intensity (center slice)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(refinement_dir / f"{channel_name}_center_slice_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return final_mask, labels


