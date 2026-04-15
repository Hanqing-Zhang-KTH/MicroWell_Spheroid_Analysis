"""
Spheroid_Profiling – 3D spheroid loading, segmentation and composition analysis

Research group:
Björn Önfelt Group
Department of Applied Physics, Division of Biophysics
Royal Institute of Technology

Coding author:
Hanqing Zhang, Researcher, Royal Institute of Technology, hanzha@kth.se, hanzha@kth.se
"""

from __future__ import annotations

import argparse
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import measure, morphology

from functions.io_utils import load_json_config, ensure_dir
from functions.data_loading import discover_samples
from functions.preprocessing import apply_optional_gaussian
from functions.thresholding import (
    fixed_global_thresholding,
    save_binary_stack,
    save_threshold_vs_manual_figures,
    save_mask_overlap_figure,
)
from functions.dl_segmentation import run_dl_segmentation_3d, run_dl_segmentation_2d
from functions.post_analysis import run_post_analysis
from functions.composition_profiling import run_composition_profiling

# Keep runtime logs clean: suppress known non-fatal library warnings.
warnings.filterwarnings(
    "ignore",
    message=r".*You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*`binary_closing` is deprecated.*",
    category=FutureWarning,
)


def _load_stack(path: Path) -> np.ndarray:
    arr = np.asarray(tifffile.imread(str(path)))
    if arr.ndim == 2:
        with tifffile.TiffFile(str(path)) as tf:
            if tf.series:
                try:
                    series_arr = np.asarray(tf.series[0].asarray())
                    if series_arr.ndim >= 3:
                        arr = series_arr
                except Exception:
                    pass
            if arr.ndim == 2 and len(tf.pages) > 1:
                first = tf.pages[0].asarray()
                first_shape = first.shape
                if all(p.shape == first_shape for p in tf.pages):
                    stacked = np.empty((len(tf.pages),) + first_shape, dtype=first.dtype)
                    stacked[0] = first
                    for i, p in enumerate(tf.pages[1:], start=1):
                        stacked[i] = p.asarray()
                    arr = stacked
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr.astype(np.float32)


def _load_binary_mask(path: Path) -> np.ndarray:
    arr = _load_stack(path)
    return arr > 0


def _manual_mask_preprocess(mask: np.ndarray, cfg: Dict[str, object], keep_largest: bool) -> np.ndarray:
    """
    Preprocess manual mask with smoothing + threshold + size filtering.
    Optionally keep only largest connected component (for nucleus).
    """
    mcfg = cfg.get("manual_mask_preprocessing", {})
    sigma = float(mcfg.get("gaussian_sigma", 1.6))
    kernel_size = max(3, int(mcfg.get("gaussian_kernel_size", 5)))
    thr = float(mcfg.get("reconstruct_threshold", 0.5))
    radius = float(mcfg.get("size_filter_radius", 3.0))
    min_size_cfg = int(mcfg.get("min_object_voxels", 0))

    truncate = max(0.5, ((kernel_size - 1) / 2.0) / max(sigma, 1e-6))
    smooth = ndi.gaussian_filter(mask.astype(np.float32), sigma=sigma, truncate=truncate)
    out = smooth >= thr

    if min_size_cfg > 0:
        min_size = min_size_cfg
    else:
        min_size = int((4.0 / 3.0) * np.pi * (radius ** 3))
    min_size = max(1, min_size)
    out = morphology.remove_small_objects(out, min_size=min_size)

    if keep_largest:
        labels = measure.label(out.astype(np.uint8), connectivity=1)
        if labels.max() > 0:
            counts = np.bincount(labels.ravel())
            largest = int(np.argmax(counts[1:]) + 1)
            out = labels == largest
    return out.astype(bool)


def _channel_post_cfg(post_cfg: Dict[str, object], channel_key: str) -> tuple[Dict[str, object], bool]:
    """
    Return per-channel post-analysis config and whether watershed refinement is enabled.
    """
    base_enabled = bool(post_cfg.get("enable_watershed_refinement", False))
    ch_enable_cfg = post_cfg.get("apply_cell_separation", post_cfg.get("channel_enable", {}))
    if isinstance(ch_enable_cfg, dict):
        enabled = bool(ch_enable_cfg.get(channel_key, base_enabled))
    else:
        enabled = base_enabled
    cfg_ch = dict(post_cfg)
    cfg_ch["enable_watershed_refinement"] = enabled
    return cfg_ch, enabled


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spheroid profiling pipeline (loading, segmentation, composition analysis)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Config/spheroid_config.json",
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--input-tiff",
        type=str,
        default="",
        help="Absolute path to a 6-channel TIFF (tumor,fibro,nucleus,tumorMask,fibroMask,nucleusMask).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="Default_Experiment",
        help="Experiment name used for results folder structure.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default="",
        help="Optional sample name override (default parsed from TIFF name suffix).",
    )
    return parser.parse_args()


def process_sample(
    sample,
    cfg: Dict[str, object],
    project_root: Path,
) -> None:
    t_sample = time.perf_counter()
    sample_tag = f"{sample.experiment_name}/{sample.name}"
    print(f"  [Sample {sample_tag}] Starting processing.")
    paths_cfg = cfg["paths"]
    rr = str(paths_cfg.get("results_root", "Results"))
    rp = Path(rr)
    results_root = ensure_dir(rp if rp.is_absolute() else (project_root / rp))

    experiment_results = ensure_dir(results_root / sample.experiment_name)
    sample_results = ensure_dir(experiment_results / sample.name)
    img_mask_dir = ensure_dir(sample_results / "ImageMask")
    img_mask_dl_dir = ensure_dir(sample_results / "ImageMaskDL")
    img_mask_mix_dir = ensure_dir(sample_results / "ImageMaskMix")
    thr_dir = ensure_dir(sample_results / "Thresholding")
    dl_dir = ensure_dir(sample_results / "DL_masks")
    refine_dir = ensure_dir(sample_results / "Refinement")
    input_dir = ensure_dir(sample_results / "Input")
    try:
        shutil.copy2(sample.tiff_path, input_dir / sample.tiff_path.name)
    except Exception:
        # Copy is best-effort (large files / permission issues).
        pass

    t_step1 = time.perf_counter()
    print(f"  [Sample {sample_tag}] Step 1/3: Loading 6-channel TIFF (raw + provided masks)...")
    stack6 = np.asarray(tifffile.imread(str(sample.tiff_path)))
    if stack6.ndim != 4 or stack6.shape[0] != 6:
        raise RuntimeError(
            f"Expected TIFF shape (6, Z, Y, X) for {sample.tiff_path}, got {stack6.shape}."
        )

    # New convention (channel-first):
    # 0 tumor raw, 1 fibro raw, 2 nucleus raw, 3 tumor mask, 4 fibro mask, 5 nucleus mask
    tumor_raw = stack6[0].astype(np.float32)
    fibro_raw = stack6[1].astype(np.float32)
    nuc_raw = stack6[2].astype(np.float32)
    tumor_man = stack6[3] > 0
    fibro_man = stack6[4] > 0
    nuc_man = stack6[5] > 0

    # Optional intensity preprocessing
    pre_cfg = cfg.get("preprocessing", {})
    nuc_raw_p = apply_optional_gaussian(nuc_raw, pre_cfg)
    tumor_raw_p = apply_optional_gaussian(tumor_raw, pre_cfg)
    fibro_raw_p = apply_optional_gaussian(fibro_raw, pre_cfg)
    print(f"  [Sample {sample_tag}] Step 1/3 done ({time.perf_counter() - t_step1:.1f}s)")

    # 1) Quantification mode: global_thresholds, dl_mode, or mix_mode
    t_step2 = time.perf_counter()
    q_cfg = cfg["quantification"]
    mode = str(q_cfg.get("mode", "global_thresholds")).strip().lower()
    # Backward compatibility: accept old "mix" and map to "mix_mode".
    if mode == "mix":
        mode = "mix_mode"
    if mode not in ("global_thresholds", "dl_mode", "mix_mode"):
        mode = "global_thresholds"
    print(f"  [Sample {sample_tag}] Step 2/3: Quantification (mode={mode}).")

    thr_cfg = q_cfg.get("global_thresholds", {})
    dl_cfg = q_cfg.get("dl_mode", {})
    mix_cfg = q_cfg.get("mix_mode", {"nucleus": "3d", "tumor": "global", "fibroblast": "global"})
    net_type = str(dl_cfg.get("network_type", "3d")).lower()
    class_fg = int(dl_cfg.get("class_label_id_foreground", 1))
    fg_val = int(dl_cfg.get("foreground_value_in_mask", 255))
    networks_root = project_root / "Networks"

    dl_overlap = float(dl_cfg.get("sliding_window_overlap", 0.25))

    def _quantify_channel(ch_name: str, raw: np.ndarray, method: str) -> np.ndarray:
        ch_display = ch_name.capitalize()
        if method == "global":
            print(f"    Quantifying {ch_display}: global threshold")
            thr_key = {"nucleus": "nucleus_draq5", "tumor": "tumor_rfp", "fibroblast": "fibroblast_gfp"}[ch_name]
            thr_val = float(thr_cfg.get(thr_key, 0))
            return (raw >= thr_val).astype(bool)
        if method == "manual":
            print(f"    Quantifying {ch_display}: provided mask + preprocessing")
            src = {"nucleus": nuc_man, "tumor": tumor_man, "fibroblast": fibro_man}[ch_name]
            keep_largest = ch_name == "nucleus"
            return _manual_mask_preprocess(src, cfg, keep_largest=keep_largest)
        if method in ("3d", "2d"):
            if method == "3d":
                out = run_dl_segmentation_3d(raw, networks_root, class_fg, fg_val, overlap=dl_overlap, channel_name=ch_display) > 0
            else:
                out = run_dl_segmentation_2d(raw, networks_root, class_fg, fg_val, channel_name=ch_display) > 0
            return out
        raise ValueError(f"Unknown method '{method}' for {ch_name}")

    masks: Dict[str, np.ndarray] = {}
    fig_dir = img_mask_dir

    if mode == "global_thresholds":
        masks = fixed_global_thresholding(
            nucleus_stack=nuc_raw_p,
            tumor_stack=tumor_raw_p,
            fibro_stack=fibro_raw_p,
            thresholds=thr_cfg,
        )
        save_binary_stack(masks["nucleus"], thr_dir / f"{sample.name}_Nucleus.tiff")
        save_binary_stack(masks["tumor"], thr_dir / f"{sample.name}_Tumor.tiff")
        save_binary_stack(masks["fibroblast"], thr_dir / f"{sample.name}_Fibro.tiff")
        for ch, man in [("nucleus", nuc_man), ("tumor", tumor_man), ("fibroblast", fibro_man)]:
            save_threshold_vs_manual_figures(masks[ch], man, fig_dir / f"{sample.name}_{ch.capitalize()}", title_prefix=ch.capitalize())
    elif mode == "dl_mode":
        masks["nucleus"] = _quantify_channel("nucleus", nuc_raw_p, net_type)
        masks["tumor"] = _quantify_channel("tumor", tumor_raw_p, net_type)
        masks["fibroblast"] = _quantify_channel("fibroblast", fibro_raw_p, net_type)
        save_binary_stack(masks["nucleus"], dl_dir / f"{sample.name}_Nucleus.tiff")
        save_binary_stack(masks["tumor"], dl_dir / f"{sample.name}_Tumor.tiff")
        save_binary_stack(masks["fibroblast"], dl_dir / f"{sample.name}_Fibro.tiff")
        fig_dir = img_mask_dl_dir
        for ch, man in [("nucleus", nuc_man), ("tumor", tumor_man), ("fibroblast", fibro_man)]:
            save_threshold_vs_manual_figures(masks[ch], man, fig_dir / f"{sample.name}_{ch.capitalize()}", title_prefix=f"{ch.capitalize()} DL")
    elif mode == "mix_mode":
        fig_dir = img_mask_mix_dir
        for ch, raw, man in [
            ("nucleus", nuc_raw_p, nuc_man),
            ("tumor", tumor_raw_p, tumor_man),
            ("fibroblast", fibro_raw_p, fibro_man),
        ]:
            meth = str(mix_cfg.get(ch, "global" if ch != "nucleus" else "3d")).lower()
            if meth in ("global", "3d", "2d", "manual"):
                pass
            else:
                meth = "3d" if ch == "nucleus" else "global"
            masks[ch] = _quantify_channel(ch, raw, meth)
            if meth == "global":
                save_binary_stack(masks[ch], thr_dir / f"{sample.name}_{ch.capitalize()}.tiff")
            elif meth in ("3d", "2d"):
                save_binary_stack(masks[ch], dl_dir / f"{sample.name}_{ch.capitalize()}.tiff")
            else:
                save_binary_stack(masks[ch], thr_dir / f"{sample.name}_{ch.capitalize()}.tiff")
            title = f"{ch.capitalize()} ({meth})"
            save_threshold_vs_manual_figures(masks[ch], man, fig_dir / f"{sample.name}_{ch.capitalize()}", title_prefix=title)

    # Overlap figures
    overlap_dir = fig_dir
    save_mask_overlap_figure(
        masks["nucleus"],
        masks["fibroblast"],
        overlap_dir / f"{sample.name}_Nucleus_Fibro_overlap.png",
        title="Overlap: nucleus vs fibroblast",
    )
    save_mask_overlap_figure(
        masks["nucleus"],
        masks["tumor"],
        overlap_dir / f"{sample.name}_Nucleus_Tumor_overlap.png",
        title="Overlap: nucleus vs tumor",
    )
    print(f"  [Sample {sample_tag}] Step 2/3 done ({time.perf_counter() - t_step2:.1f}s)")

    t_step3 = time.perf_counter()
    print(f"  [Sample {sample_tag}] Step 3/3: Post-analysis and composition profiling...")
    post_cfg = cfg.get("post_analysis", {})

    # Part 1/4: Post-analysis (Nucleus)
    t0 = time.perf_counter()
    print(f"    [3.1/4] Post-analysis: Nucleus...", end=" ", flush=True)
    nuc_post_cfg, nuc_ws_enabled = _channel_post_cfg(post_cfg, "nucleus")
    if nuc_ws_enabled:
        nuc_ref, nuc_labels = run_post_analysis(
            intensity_stack=nuc_raw_p,
            binary_mask=masks["nucleus"],
            cfg=nuc_post_cfg,
            refinement_dir=refine_dir,
            channel_name="Nucleus",
        )
    else:
        nuc_ref, _ = run_post_analysis(
            intensity_stack=nuc_raw_p,
            binary_mask=masks["nucleus"],
            cfg=nuc_post_cfg,
            refinement_dir=refine_dir,
            channel_name="Nucleus",
        )
        nuc_labels = None
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    # Part 2/4: Post-analysis (Tumor)
    t0 = time.perf_counter()
    print(f"    [3.2/4] Post-analysis: Tumor...", end=" ", flush=True)
    tumor_post_cfg, tumor_ws_enabled = _channel_post_cfg(post_cfg, "tumor")
    if tumor_ws_enabled:
        tumor_ref, tumor_labels = run_post_analysis(
            intensity_stack=tumor_raw_p,
            binary_mask=masks["tumor"],
            cfg=tumor_post_cfg,
            refinement_dir=refine_dir,
            channel_name="Tumor",
        )
    else:
        tumor_ref, _ = run_post_analysis(
            intensity_stack=tumor_raw_p,
            binary_mask=masks["tumor"],
            cfg=tumor_post_cfg,
            refinement_dir=refine_dir,
            channel_name="Tumor",
        )
        tumor_labels = None
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    # Part 3/4: Post-analysis (Fibroblast)
    t0 = time.perf_counter()
    print(f"    [3.3/4] Post-analysis: Fibroblast...", end=" ", flush=True)
    fibro_post_cfg, fibro_ws_enabled = _channel_post_cfg(post_cfg, "fibroblast")
    if fibro_ws_enabled:
        fibro_ref, fibro_labels = run_post_analysis(
            intensity_stack=fibro_raw_p,
            binary_mask=masks["fibroblast"],
            cfg=fibro_post_cfg,
            refinement_dir=refine_dir,
            channel_name="Fibroblast",
        )
    else:
        fibro_ref, _ = run_post_analysis(
            intensity_stack=fibro_raw_p,
            binary_mask=masks["fibroblast"],
            cfg=fibro_post_cfg,
            refinement_dir=refine_dir,
            channel_name="Fibroblast",
        )
        fibro_labels = None
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    # Part 4/4: Composition profiling
    t0 = time.perf_counter()
    print(f"    [3.4/4] Composition profiling (radial tables, stats, plots)...", end=" ", flush=True)
    comp_cfg = cfg["composition_profiling"]
    instance_labels_for_xlsx = {}
    if nuc_labels is not None:
        instance_labels_for_xlsx["nucleus"] = nuc_labels
    if tumor_labels is not None:
        instance_labels_for_xlsx["tumor"] = tumor_labels
    if fibro_labels is not None:
        instance_labels_for_xlsx["fibroblast"] = fibro_labels
    run_composition_profiling(
        sample_name=sample.name,
        nucleus_mask=nuc_ref,
        tumor_mask=tumor_ref,
        fibro_mask=fibro_ref,
        cfg=comp_cfg,
        out_dir=sample_results,
        instance_labels=instance_labels_for_xlsx,
        post_cfg=post_cfg,
    )
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    print(f"  [Sample {sample_tag}] Step 3/3 done ({time.perf_counter() - t_step3:.1f}s)")
    print(f"  [Sample {sample_tag}] Completed ({time.perf_counter() - t_sample:.1f}s total).")


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent
    cfg_path = (project_root / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_json_config(cfg_path)

    if not args.input_tiff:
        raise RuntimeError("Missing --input-tiff. Provide an absolute path to a 6-channel TIFF.")

    tiff_path = Path(args.input_tiff)
    if not tiff_path.exists():
        raise FileNotFoundError(f"Input TIFF not found: {tiff_path}")

    exp_name = str(args.experiment).strip() or "Default_Experiment"
    sample_name = str(args.sample).strip()
    if not sample_name:
        stem = tiff_path.stem
        if "_" not in stem:
            raise RuntimeError(f"Cannot parse sample name from TIFF (missing '_<suffix>'): {tiff_path.name}")
        sample_name = stem.rsplit("_", 1)[-1].strip()
        if not sample_name:
            raise RuntimeError(f"Cannot parse non-empty sample name suffix from TIFF: {tiff_path.name}")

    sample = type("Sample", (), {})()
    sample.experiment_name = exp_name
    sample.name = sample_name
    sample.tiff_path = tiff_path

    print("Found 1 spheroid sample (single TIFF mode).")
    print(f"[1/1] Processing sample: {sample.experiment_name}/{sample.name}")
    process_sample(sample, cfg, project_root)


if __name__ == "__main__":
    main()

