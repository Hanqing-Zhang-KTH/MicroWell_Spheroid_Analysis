"""
Averaging pattern analysis for selected result samples.

Research group:
Björn Önfelt Group
Department of Applied Physics, Division of Biophysics
Royal Institute of Technology

Coding author:
Hanqing Zhang, Researcher, Royal Institute of Technology, hanzha@kth.se
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import measure

from functions.io_utils import load_json_config, ensure_dir
from functions.post_analysis import run_post_analysis
from functions.composition_profiling import run_composition_profiling


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Averaging pattern analysis from selected result samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="Config/spheroid_config.json")
    parser.add_argument(
        "--channel",
        type=str,
        default="nucleus",
        choices=["nucleus", "tumor", "fibroblast"],
        help="Channel used for alignment centroid.",
    )
    parser.add_argument(
        "--sample-dir",
        action="append",
        default=[],
        help="Result sample directory path. Can be provided multiple times.",
    )
    return parser.parse_args()


def _largest_component_centroid(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask.astype(np.uint8), connectivity=1)
    if labels.max() <= 0:
        raise RuntimeError("No foreground object found in alignment mask.")
    counts = np.bincount(labels.ravel())
    largest = int(np.argmax(counts[1:]) + 1)
    coords = np.argwhere(labels == largest)
    return coords.mean(axis=0)  # z, y, x


def _read_tiff_stack_robust(path: Path) -> np.ndarray:
    """
    Read TIFF robustly:
    - use tifffile.imread result when already >=3D
    - if 2D but file has multiple equal-size pages, stack pages into 3D (Z, Y, X)
    """
    try:
        arr = np.asarray(tifffile.imread(str(path)))
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF: {path} ({exc})") from exc
    if arr.ndim >= 3:
        return arr
    with tifffile.TiffFile(str(path)) as tf:
        if tf.series:
            try:
                series_arr = np.asarray(tf.series[0].asarray())
                if series_arr.ndim >= 3:
                    return series_arr
            except Exception:
                pass
        if len(tf.pages) > 1:
            first = tf.pages[0].asarray()
            first_shape = first.shape
            for p in tf.pages[1:]:
                if p.shape != first_shape:
                    return arr
            out = np.empty((len(tf.pages),) + first_shape, dtype=first.dtype)
            out[0] = first
            for i, p in enumerate(tf.pages[1:], start=1):
                out[i] = p.asarray()
            return out
    return arr


def _find_input_tiff(sample_dir: Path) -> Path:
    input_dir = sample_dir / "Input"
    candidates: list[Path] = []
    if input_dir.exists():
        candidates.extend(sorted(input_dir.glob("*.tif")))
        candidates.extend(sorted(input_dir.glob("*.tiff")))
    if not candidates:
        candidates.extend(sorted(sample_dir.glob("*.tif")))
        candidates.extend(sorted(sample_dir.glob("*.tiff")))
    if not candidates:
        raise FileNotFoundError(f"No input TIFF found under {sample_dir}/Input (or sample root).")
    return candidates[0]


def _load_6ch_tiff(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arr = np.asarray(tifffile.imread(str(path)))
    if arr.ndim != 4 or arr.shape[0] != 6:
        raise RuntimeError(f"Expected input TIFF shape (6, Z, Y, X), got {arr.shape} for {path}")
    raw = {
        "tumor": arr[0].astype(np.float32),
        "fibroblast": arr[1].astype(np.float32),
        "nucleus": arr[2].astype(np.float32),
    }
    masks = {
        "tumor": (arr[3] > 0),
        "fibroblast": (arr[4] > 0),
        "nucleus": (arr[5] > 0),
    }
    return raw, masks


def _shift_stack(stack: np.ndarray, shift_zyx: np.ndarray, order: int) -> np.ndarray:
    return ndi.shift(
        stack.astype(np.float32),
        shift=shift_zyx,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _step2_mask_candidates(sample_dir: Path, sample_name: str, channel_key: str) -> list[Path]:
    ch_cap = channel_key.capitalize()
    cands = [
        sample_dir / "Thresholding" / f"{sample_name}_{ch_cap}.tiff",
        sample_dir / "Thresholding" / f"{sample_name}_{ch_cap}.tif",
        sample_dir / "DL_masks" / f"{sample_name}_{ch_cap}.tiff",
        sample_dir / "DL_masks" / f"{sample_name}_{ch_cap}.tif",
    ]
    if channel_key == "fibroblast":
        cands.extend(
            [
                sample_dir / "Thresholding" / f"{sample_name}_Fibro.tiff",
                sample_dir / "Thresholding" / f"{sample_name}_Fibro.tif",
                sample_dir / "DL_masks" / f"{sample_name}_Fibro.tiff",
                sample_dir / "DL_masks" / f"{sample_name}_Fibro.tif",
            ]
        )
    return cands


def _find_step2_mask(sample_dir: Path, sample_name: str, channel_key: str) -> Path:
    for p in _step2_mask_candidates(sample_dir, sample_name, channel_key):
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Step-2 mask not found for {channel_key} in {sample_dir}."
    )


def _load_mask_matching_raw_shape(
    sample_dir: Path,
    sample_name: str,
    channel_key: str,
    raw_shape: tuple[int, ...],
) -> tuple[np.ndarray, Path]:
    """
    Load a step-2 mask that matches the raw stack shape.
    Tries all candidates and returns the first shape-compatible one.
    """
    checked: list[str] = []
    for p in _step2_mask_candidates(sample_dir, sample_name, channel_key):
        if not p.exists():
            continue
        arr = _read_tiff_stack_robust(p)
        checked.append(f"{p.name}:{arr.shape}")
        if arr.shape == raw_shape:
            return (arr > 0), p
        # Allow 2D mask only for single-slice raw stacks.
        if arr.ndim == 2 and len(raw_shape) == 3 and raw_shape[0] == 1 and arr.shape == raw_shape[1:]:
            return (arr[None, ...] > 0), p
    raise RuntimeError(
        f"No shape-matching step-2 mask for {channel_key} in {sample_dir}. "
        f"Raw shape={raw_shape}. Checked: {', '.join(checked) if checked else 'no candidate files found'}"
    )


def _isovolume(raw: np.ndarray, mask: np.ndarray, voxel_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    vx = float(voxel_cfg.get("x", 1.0))
    vy = float(voxel_cfg.get("y", 1.0))
    vz = float(voxel_cfg.get("z", 1.0))
    target = min(vx, vy, vz)
    zoom = (vz / target, vy / target, vx / target)  # z, y, x
    raw_iso = ndi.zoom(raw.astype(np.float32), zoom=zoom, order=1)
    mask_iso = ndi.zoom(mask.astype(np.float32), zoom=zoom, order=0) > 0.5
    return raw_iso.astype(np.float32), mask_iso.astype(bool)


def _channel_post_cfg(post_cfg: dict, channel_key: str) -> tuple[dict, bool]:
    """Per-channel enable switch for averaging post-analysis."""
    base_enabled = bool(post_cfg.get("enable_watershed_refinement", False))
    ch_enable_cfg = post_cfg.get("apply_cell_separation", post_cfg.get("channel_enable", {}))
    if isinstance(ch_enable_cfg, dict):
        enabled = bool(ch_enable_cfg.get(channel_key, base_enabled))
    else:
        enabled = base_enabled
    cfg_ch = dict(post_cfg)
    cfg_ch["enable_watershed_refinement"] = enabled
    return cfg_ch, enabled


def main() -> None:
    args = _parse_args()
    sample_dirs = [Path(p) for p in args.sample_dir]
    if not sample_dirs:
        raise RuntimeError("No --sample-dir provided.")

    project_root = Path(__file__).resolve().parent
    cfg_path = (project_root / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_json_config(cfg_path)
    paths_cfg = cfg["paths"]
    rr = str(paths_cfg.get("results_root", "Results"))
    rp = Path(rr)
    results_root = (rp if rp.is_absolute() else (project_root / rp)).resolve()

    channel = args.channel.lower()
    channels = ["nucleus", "tumor", "fibroblast"]

    raw_sums: dict[str, np.ndarray] = {}
    mask_counts: dict[str, np.ndarray] = {}
    template_shape: tuple[int, int, int] | None = None
    rows: list[dict] = []
    used = 0

    print(f"[Averaging] Alignment channel: {channel}")
    for i, sample_dir in enumerate(sample_dirs, start=1):
        sample_dir = sample_dir.resolve()
        rel = sample_dir.relative_to(results_root.resolve())
        if len(rel.parts) < 2:
            raise RuntimeError(f"Result sample path must be Results/<experiment>/<sample>: {sample_dir}")
        experiment_name, sample_name = rel.parts[0], rel.parts[1]
        input_tiff = _find_input_tiff(sample_dir)
        raw_ch, mask_ch = _load_6ch_tiff(input_tiff)
        raw_align = raw_ch[channel]
        mask_align = mask_ch[channel]
        if template_shape is None:
            template_shape = raw_align.shape
        elif raw_align.shape != template_shape or mask_align.shape != template_shape:
            raise RuntimeError(
                f"Sample shape mismatch. Template {template_shape}, got {raw_align.shape} "
                f"for {experiment_name}/{sample_name}."
            )

        centroid = _largest_component_centroid(mask_align)
        target_center = (np.array(template_shape, dtype=np.float32) - 1.0) / 2.0
        shift = target_center - centroid

        print(f"[Averaging] [{i}/{len(sample_dirs)}] {experiment_name}/{sample_name} shift={shift.round(3)}")

        for ch in channels:
            raw = raw_ch[ch]
            mask = mask_ch[ch]

            if raw.shape != template_shape or mask.shape != template_shape:
                raise RuntimeError(
                    f"Shape mismatch for {experiment_name}/{sample_name}, channel {ch}. "
                    f"Expected {template_shape}, raw {raw.shape}, mask {mask.shape}."
                )

            raw_aligned = _shift_stack(raw, shift, order=1)
            mask_aligned = _shift_stack(mask.astype(np.float32), shift, order=0) > 0.5

            if ch not in raw_sums:
                raw_sums[ch] = np.zeros(template_shape, dtype=np.float64)
                mask_counts[ch] = np.zeros(template_shape, dtype=np.uint16)
            raw_sums[ch] += raw_aligned
            mask_counts[ch] += mask_aligned.astype(np.uint16)

        rows.append(
            {
                "Experiment": experiment_name,
                "Sample": sample_name,
                "AlignmentChannel": channel,
                "ShiftZ": float(shift[0]),
                "ShiftY": float(shift[1]),
                "ShiftX": float(shift[2]),
            }
        )
        used += 1

    if used == 0:
        raise RuntimeError("No samples processed for averaging.")

    threshold_count = int(np.ceil(0.5 * used))
    avg_raw: dict[str, np.ndarray] = {}
    avg_mask: dict[str, np.ndarray] = {}
    for ch in channels:
        avg_raw[ch] = (raw_sums[ch] / float(used)).astype(np.float32)
        avg_mask[ch] = (mask_counts[ch] >= threshold_count).astype(bool)

    voxel_cfg = cfg.get("composition_profiling", {}).get("voxel_size_um", {})
    avg_raw_iso: dict[str, np.ndarray] = {}
    avg_mask_iso: dict[str, np.ndarray] = {}
    for ch in channels:
        r_iso, m_iso = _isovolume(avg_raw[ch], avg_mask[ch], voxel_cfg)
        avg_raw_iso[ch] = r_iso
        avg_mask_iso[ch] = m_iso

    out_root = ensure_dir(results_root / f"Averaging_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    patterns_dir = ensure_dir(out_root / "Patterns")
    seg_dir = ensure_dir(patterns_dir / "Segmentation")

    for ch in channels:
        ch_cap = ch.capitalize()
        tifffile.imwrite(str(patterns_dir / f"Averaged_{ch_cap}_raw.tiff"), avg_raw_iso[ch].astype(np.float32))
        tifffile.imwrite(str(seg_dir / f"Averaged_{ch_cap}_mask.tiff"), (avg_mask_iso[ch].astype(np.uint8) * 255))

    df = pd.DataFrame(rows)
    df.to_excel(out_root / "AveragingSelection.xlsx", index=False)
    with (out_root / "AveragingSelection.txt").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['Experiment']}/{r['Sample']}\n")
        f.write(f"\nAlignmentChannel: {channel}\n")

    post_cfg = dict(cfg.get("post_analysis", {}))
    avg_cfg = post_cfg.get("averaging", {})
    if isinstance(avg_cfg, dict):
        avg_refine = avg_cfg.get(
            "enable_watershed_refinement",
            post_cfg.get("enable_watershed_refinement_averaging", True),
        )
    else:
        avg_refine = post_cfg.get("enable_watershed_refinement_averaging", True)
    post_cfg["enable_watershed_refinement"] = bool(avg_refine)

    if isinstance(avg_cfg, dict):
        avg_ch = avg_cfg.get("apply_cell_separation", avg_cfg.get("channel_enable", None))
        if isinstance(avg_ch, dict):
            post_cfg["apply_cell_separation"] = dict(avg_ch)
    refinement_dir = ensure_dir(out_root / "Refinement")

    instance_labels = {}
    for ch in channels:
        ch_cap = ch.capitalize()
        ch_post_cfg, ch_enabled = _channel_post_cfg(post_cfg, ch)
        if ch_enabled:
            ref_mask, labels = run_post_analysis(
                intensity_stack=avg_raw_iso[ch],
                binary_mask=avg_mask_iso[ch],
                cfg=ch_post_cfg,
                refinement_dir=refinement_dir,
                channel_name=f"Averaged_{ch_cap}",
            )
            avg_mask_iso[ch] = ref_mask
            instance_labels[ch] = labels
        else:
            ref_mask, _ = run_post_analysis(
                intensity_stack=avg_raw_iso[ch],
                binary_mask=avg_mask_iso[ch],
                cfg=ch_post_cfg,
                refinement_dir=refinement_dir,
                channel_name=f"Averaged_{ch_cap}",
            )
            avg_mask_iso[ch] = ref_mask

    run_composition_profiling(
        sample_name=f"Averaging_{channel}",
        nucleus_mask=avg_mask_iso["nucleus"],
        tumor_mask=avg_mask_iso["tumor"],
        fibro_mask=avg_mask_iso["fibroblast"],
        cfg=cfg.get("composition_profiling", {}),
        out_dir=out_root,
        instance_labels=instance_labels,
        post_cfg=post_cfg,
    )

    print(f"[Averaging] Completed. Output: {out_root}")


if __name__ == "__main__":
    main()

