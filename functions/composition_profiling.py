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
from typing import Dict, Optional, Tuple

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.morphology import convex_hull_image
import pandas as pd
import matplotlib.pyplot as plt

from .io_utils import ensure_dir


def _ensure_bool(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(bool)


def _convex_spheroid_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Build a convex approximation of the largest connected component in the spheroid mask.
    This is a simplified 3D analogue of the MATLAB convex-hull regularization.
    """
    mask = _ensure_bool(mask)
    labeled, num = ndi.label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    largest = int(np.argmax(counts[1:]) + 1)
    sph = labeled == largest

    # Coarse morphological closing to regularize
    se = morphology.ball(3)
    sph = morphology.closing(sph, se)

    # Use binary fill holes slice-wise as a practical convex proxy
    out = np.zeros_like(sph, dtype=bool)
    for z in range(sph.shape[0]):
        out[z] = ndi.binary_fill_holes(sph[z])
    return out


def _distance_shells(convex_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute integer distance shells from the spheroid surface in voxels
    (1-voxel-thick shells inside the convex volume).
    """
    convex_mask = _ensure_bool(convex_mask)
    if not np.any(convex_mask):
        return np.zeros_like(convex_mask, dtype=int), 0
    surface = ndi.binary_dilation(convex_mask) ^ ndi.binary_erosion(convex_mask)
    dist = ndi.distance_transform_edt(convex_mask)
    dist[~convex_mask] = 0.0
    shells = np.round(dist).astype(int)
    level_num = int(shells.max())
    return shells, level_num


def _build_radial_tables(
    shells: np.ndarray,
    level_num: int,
    mask_A: np.ndarray,
    mask_B: np.ndarray,
    voxel_size_xy_um: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-shell and equal-volume composition tables similar to the MATLAB
    SingleSphroidinfiltrateAnalysis implementation, but generalized for two
    binary masks A and B.
    """
    shells = shells.astype(int)
    mask_A = _ensure_bool(mask_A)
    mask_B = _ensure_bool(mask_B)
    if level_num == 0:
        return np.zeros((6, 0), dtype=float), np.zeros((6, 0), dtype=float)

    x_um = (np.arange(level_num) + 1) * float(voxel_size_xy_um)

    # 1 distance; 2 shell voxels; 3 empty; 4 A-only; 5 B-only; 6 A∩B
    # Vectorized: encoding = shell*4 + A*2 + B gives 8 bins per shell (0..7)
    shell_flat = shells.ravel().astype(np.int64)
    mask_A_flat = mask_A.ravel().astype(np.int64)
    mask_B_flat = mask_B.ravel().astype(np.int64)
    encoding = np.where(
        shell_flat > 0,
        shell_flat * 4 + mask_A_flat * 2 + mask_B_flat,
        -1,
    )
    valid = encoding >= 0
    hist = np.bincount(encoding[valid], minlength=4 * level_num + 4)
    dist_table = np.zeros((6, level_num), dtype=float)
    dist_table[0, :] = x_um
    for d in range(1, level_num + 1):
        base = 4 * d
        n_empty = hist[base + 0]
        n_B_only = hist[base + 1]
        n_A_only = hist[base + 2]
        n_overlap = hist[base + 3]
        n_shell = n_empty + n_B_only + n_A_only + n_overlap
        dist_table[1:6, d - 1] = [n_shell, n_empty, n_A_only, n_B_only, n_overlap]

    shell_vox = dist_table[1, :]
    cum_shell = np.cumsum(shell_vox)
    total_vol = cum_shell[-1] if cum_shell.size > 0 else 0.0
    if total_vol <= 0:
        return dist_table, np.zeros((6, 0), dtype=float)

    # Equal-volume partitions
    num_parts = max(1, min(level_num, 10))
    vol_per_part = total_vol / float(num_parts)
    part_idx = []
    acc = 0.0
    for _ in range(num_parts):
        acc += vol_per_part
        idx = int(np.argmin(np.abs(cum_shell - acc))) + 1
        part_idx.append(idx)
    part_idx = sorted(set(max(1, min(level_num, i)) for i in part_idx))

    cum_A = np.cumsum(dist_table[3, :])
    cum_B = np.cumsum(dist_table[4, :])
    cum_overlap = np.cumsum(dist_table[5, :])
    cum_empty = np.cumsum(dist_table[2, :])

    part_shell = [cum_shell[part_idx[0] - 1]]
    part_empty = [cum_empty[part_idx[0] - 1]]
    part_A = [cum_A[part_idx[0] - 1]]
    part_B = [cum_B[part_idx[0] - 1]]
    part_overlap = [cum_overlap[part_idx[0] - 1]]
    for i in range(1, len(part_idx)):
        part_shell.append(cum_shell[part_idx[i] - 1] - cum_shell[part_idx[i - 1] - 1])
        part_empty.append(cum_empty[part_idx[i] - 1] - cum_empty[part_idx[i - 1] - 1])
        part_A.append(cum_A[part_idx[i] - 1] - cum_A[part_idx[i - 1] - 1])
        part_B.append(cum_B[part_idx[i] - 1] - cum_B[part_idx[i - 1] - 1])
        part_overlap.append(cum_overlap[part_idx[i] - 1] - cum_overlap[part_idx[i - 1] - 1])

    part_dist_um = x_um[np.array(part_idx) - 1]

    eqv = np.zeros((6, len(part_idx)), dtype=float)
    eqv[0, :] = part_dist_um
    eqv[1, :] = part_shell
    eqv[2, :] = part_empty
    eqv[3, :] = part_A
    eqv[4, :] = part_B
    eqv[5, :] = part_overlap

    return dist_table, eqv


def _plot_composition_from_table(
    table: np.ndarray,
    title: str,
    x_label: str,
    out_path: Path,
    type_a_label: str = "tumor",
) -> None:
    """
    Create a stacked composition plot from a 6xN table:
      row 0: distance (unused here, we use step index on x-axis)
      row 1: shell/partition voxels
      row 2: empty voxels
      row 3: type A voxels
      row 4: type B voxels
      row 5: overlap voxels
    All components are normalised so that A + B + overlap + empty = 1 per step.
    """
    if table.shape[1] == 0:
        return

    # x-axis in physical units (µm) from row 0
    x = table[0, :]
    if np.all(x == 0):
        x = np.arange(1, table.shape[1] + 1, dtype=float)

    shell = table[1, :]
    empty = table[2, :]
    A = table[3, :]
    B = table[4, :]
    overlap = table[5, :]

    # Avoid division by zero: where shell == 0, set proportions to 0
    with np.errstate(divide="ignore", invalid="ignore"):
        total = np.where(shell > 0, shell, 1.0)
        prop_empty = np.where(shell > 0, empty / total, 0.0)
        prop_A = np.where(shell > 0, A / total, 0.0)
        prop_B = np.where(shell > 0, B / total, 0.0)
        prop_overlap = np.where(shell > 0, overlap / total, 0.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    # Compute bin edges so neighbouring bars share boundaries exactly and
    # there are no white gaps; then draw bars from the left edge with
    # align="edge".
    if x.size > 1:
        edges = np.zeros(x.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - (edges[1] - x[0])
        edges[-1] = x[-1] + (x[-1] - edges[-2])
        lefts = edges[:-1]
        widths = edges[1:] - edges[:-1]
    else:
        lefts = x.copy()
        widths = np.array([1.0], dtype=float)

    bottom = np.zeros_like(x, dtype=float)

    type_a_str = str(type_a_label).lower()
    type_a_display = "nucleus" if type_a_str == "nucleus" else "tumor"
    plt.bar(lefts, prop_A, bottom=bottom, color="#d62728", label=f"Type A ({type_a_display})", width=widths, align="edge")
    bottom += prop_A
    plt.bar(lefts, prop_B, bottom=bottom, color="#1f77b4", label="Type B (fibroblast)", width=widths, align="edge")
    bottom += prop_B
    plt.bar(lefts, prop_overlap, bottom=bottom, color="#9467bd", label="Overlap A∩B", width=widths, align="edge")
    bottom += prop_overlap
    plt.bar(lefts, prop_empty, bottom=bottom, color="#7f7f7f", label="Empty", width=widths, align="edge")

    plt.xlabel(x_label)
    plt.ylabel("Composition (fraction of voxels)")
    plt.title(title)
    plt.ylim(0, 1.05)
    if x.size > 0:
        left_lim = lefts[0]
        right_lim = lefts[-1] + widths[-1]
        plt.xlim(left_lim, right_lim)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()


def _plot_smoothed_typeA_curves(
    table: np.ndarray,
    title: str,
    x_label: str,
    out_path: Path,
    type_a_label: str,
    fine_steps: int = 400,
    smooth_sigma_bins: float = 2.0,
) -> None:
    """
    Plot two smoothed curves on black background:
    - TypeA fraction
    - Non-TypeA fraction (TypeB + Overlap)
    Empty is intentionally not shown.
    """
    if table.shape[1] == 0:
        return

    x = table[0, :]
    if np.all(x == 0):
        x = np.arange(1, table.shape[1] + 1, dtype=float)

    shell = table[1, :]
    A = table[3, :]
    B = table[4, :]
    overlap = table[5, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        total = np.where(shell > 0, shell, 1.0)
        prop_A = np.where(shell > 0, A / total, 0.0)
        prop_nonA = np.where(shell > 0, (B + overlap) / total, 0.0)

    if x.size < 2:
        x_fine = x.copy()
        A_fine = prop_A.copy()
        nonA_fine = prop_nonA.copy()
    else:
        fine_steps = max(50, int(fine_steps))
        x_fine = np.linspace(float(x.min()), float(x.max()), fine_steps, dtype=np.float32)
        A_fine = np.interp(x_fine, x.astype(np.float32), prop_A.astype(np.float32))
        nonA_fine = np.interp(x_fine, x.astype(np.float32), prop_nonA.astype(np.float32))

    # Smooth in x-index space (bin units)
    sigma = max(0.0, float(smooth_sigma_bins))
    if sigma > 0 and x_fine.size >= 5:
        A_s = ndi.gaussian_filter1d(A_fine.astype(np.float32), sigma=sigma, mode="nearest")
        nonA_s = ndi.gaussian_filter1d(nonA_fine.astype(np.float32), sigma=sigma, mode="nearest")
    else:
        A_s = A_fine
        nonA_s = nonA_fine

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor="black")
    ax.set_facecolor("black")

    type_a_str = str(type_a_label).lower()
    type_a_display = "nucleus" if type_a_str == "nucleus" else "tumor"
    ax.plot(x_fine, np.clip(A_s, 0, 1), color="#ff4d4d", linewidth=2.0, label=f"Type A ({type_a_display}) – smoothed")
    ax.plot(x_fine, np.clip(nonA_s, 0, 1), color="#4da6ff", linewidth=2.0, label="Non-TypeA (Type B + Overlap) – smoothed")

    ax.set_title(title, color="white")
    ax.set_xlabel(x_label, color="white")
    ax.set_ylabel("Composition (fraction)", color="white")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(float(x_fine.min()), float(x_fine.max()))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(True, color="white", alpha=0.15, linewidth=0.6)
    leg = ax.legend(loc="upper right", frameon=True)
    leg.get_frame().set_facecolor("black")
    leg.get_frame().set_edgecolor("white")
    for t in leg.get_texts():
        t.set_color("white")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_detected_cells_sheet(
    labels: np.ndarray,
    channel_name: str,
    single_cell_voxel_estimate: float,
    voxel_size_um: Dict[str, float],
) -> Optional[pd.DataFrame]:
    """Build DetectedCells table from instance labels when refinement was used."""
    if labels is None or labels.max() == 0:
        return None
    vx = float(voxel_size_um.get("x", 1.0))
    vy = float(voxel_size_um.get("y", 1.0))
    vz = float(voxel_size_um.get("z", 1.0))
    props = measure.regionprops(labels.astype(np.int32))
    rows = []
    for i, p in enumerate(props, start=1):
        if p.label == 0:
            continue
        centroid = p.centroid  # (z, y, x) in skimage
        vol = int(p.area)  # volume in voxels for 3D
        est_cells = max(1, int(vol / single_cell_voxel_estimate))
        # Bounding box: (min_row, min_col, min_slice, max_row, max_col, max_slice)
        bbox = p.bbox
        if len(bbox) >= 6:
            maj_x = bbox[4] - bbox[1]  # col extent
            maj_y = bbox[3] - bbox[0]  # row extent
            maj_z = bbox[5] - bbox[2]  # slice extent
        else:
            maj_x = maj_y = maj_z = 0
        # centroid: (row, col, slice) = (y, x, z) -> output (x, y, z) in voxel coords
        cx, cy, cz = centroid[1], centroid[0], centroid[2]
        rows.append({
            "id": i,
            "center_x": cx,
            "center_y": cy,
            "center_z": cz,
            "voxels": vol,
            "major_axis_x": maj_x,
            "major_axis_y": maj_y,
            "major_axis_z": maj_z,
            "estimated_cell_number": est_cells,
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


def run_composition_profiling(
    sample_name: str,
    nucleus_mask: np.ndarray,
    tumor_mask: np.ndarray,
    fibro_mask: np.ndarray,
    cfg: Dict[str, object],
    out_dir: Path,
    instance_labels: Optional[Dict[str, np.ndarray]] = None,
    post_cfg: Optional[Dict[str, object]] = None,
) -> None:
    """
    Re-implement the core ideas of SingleSphroidinfiltrateAnalysis.m for three
    binary masks (nucleus, tumor, fibroblast).

    Spheroid mask is taken from nucleus by default, then a convex approximation
    is built. Cell-type A and B masks are selected by name and used to compute
    per-shell and equal-volume radial statistics, saved as an xlsx file.
    """
    out_dir = ensure_dir(out_dir)
    dist_plots_dir = ensure_dir(out_dir / "DistributionPlots")
    voxel = cfg.get("voxel_size_um", {})
    vx = float(voxel.get("x", 1.0))
    vy = float(voxel.get("y", 1.0))
    vz = float(voxel.get("z", 1.0))
    voxel_xy = (vx + vy) / 2.0

    spheroid_channel = str(cfg.get("spheroid_mask_channel", "nucleus")).lower()
    if spheroid_channel == "nucleus":
        sph_mask = nucleus_mask
    elif spheroid_channel == "tumor":
        sph_mask = tumor_mask
    else:
        sph_mask = fibro_mask

    convex = _convex_spheroid_from_mask(sph_mask)
    shells, level_num = _distance_shells(convex)

    # Two analyses: Type A = tumor, Type A = nucleus; Type B always fibroblast.
    variants = [
        ("Tumor", tumor_mask),
        ("Nucleus", nucleus_mask),
    ]

    xlsx_path = out_dir / f"{sample_name}_CompositionStat.xlsx"
    headers = [
        "Contour2Center Distance (um)",
        "Shell voxels (#)",
        "Empty voxels (#)",
        "Type A voxels (#)",
        "Type B voxels (#)",
        "Overlap A∩B voxels (#)",
    ]
    voxel = cfg.get("voxel_size_um", {})
    single_cell_vol = float(cfg.get("single_cell_voxel_estimate", 6000))
    ri_correction = float(cfg.get("refractive_index_correction", 1.0))

    # Use pandas' default engine to avoid hard dependency on xlsxwriter.
    with pd.ExcelWriter(xlsx_path) as writer:
        # Config / metadata sheet
        config_rows = [
            ["Parameter", "Value"],
            ["single_cell_voxel_estimate", single_cell_vol],
            ["refractive_index_correction", ri_correction],
            ["voxel_size_um_x", voxel.get("x", "")],
            ["voxel_size_um_y", voxel.get("y", "")],
            ["voxel_size_um_z", voxel.get("z", "")],
        ]
        if post_cfg:
            config_rows.extend([
                ["enable_watershed_refinement", post_cfg.get("enable_watershed_refinement", "")],
                ["min_cell_volume_voxels", post_cfg.get("min_cell_volume_voxels", "")],
            ])
        pd.DataFrame(config_rows[1:], columns=config_rows[0]).to_excel(writer, sheet_name="Config", index=False)

        # DetectedCells sheets for channels where instance labels are provided.
        if instance_labels:
            for ch_name, labels in instance_labels.items():
                df_cells = _build_detected_cells_sheet(labels, ch_name, single_cell_vol, voxel)
                if df_cells is not None and len(df_cells) > 0:
                    df_cells.to_excel(writer, sheet_name=f"DetectedCells_{ch_name.capitalize()}", index=False)
        for label, mask_A in variants:
            mask_B = fibro_mask
            dist_table, eqv_table = _build_radial_tables(shells, level_num, mask_A, mask_B, voxel_xy)

            df_shell = pd.DataFrame({headers[i]: dist_table[i, :] for i in range(dist_table.shape[0])})
            df_eqv = pd.DataFrame({headers[i]: eqv_table[i, :] for i in range(eqv_table.shape[0])})

            # Sheet names with variant suffix
            df_shell.to_excel(writer, sheet_name=f"Distributions_{label}", index=False)
            df_eqv.to_excel(writer, sheet_name=f"EquiVolumeDistribution_{label}", index=False)

            # 3-region group summaries
            def _three_group_summary(table: np.ndarray, sheet_name: str) -> None:
                if table.shape[1] == 0:
                    return
                n_steps = table.shape[1]
                g1 = n_steps // 3
                g2 = n_steps // 3
                g3 = n_steps - g1 - g2
                idx1 = np.arange(0, g1)
                idx2 = np.arange(g1, g1 + g2)
                idx3 = np.arange(g1 + g2, n_steps)

                groups = [idx1, idx2, idx3]
                rows = []
                for gi, idx in enumerate(groups, start=1):
                    if idx.size == 0:
                        continue
                    shell_sum = float(table[1, idx].sum())
                    empty_sum = float(table[2, idx].sum())
                    A_sum = float(table[3, idx].sum())
                    B_sum = float(table[4, idx].sum())
                    overlap_sum = float(table[5, idx].sum())
                    total = max(shell_sum, 1.0)
                    rows.append(
                        {
                            "Group": gi,
                            "StartStepIndex": int(idx[0] + 1),
                            "EndStepIndex": int(idx[-1] + 1),
                            "ShellVoxels": shell_sum,
                            "EmptyVoxels": empty_sum,
                            "TypeA_Voxels": A_sum,
                            "TypeB_Voxels": B_sum,
                            "Overlap_Voxels": overlap_sum,
                            "Empty_Fraction": empty_sum / total,
                            "TypeA_Fraction": A_sum / total,
                            "TypeB_Fraction": B_sum / total,
                            "Overlap_Fraction": overlap_sum / total,
                        }
                    )
                if rows:
                    df_g = pd.DataFrame(rows)
                    df_g.to_excel(writer, sheet_name=sheet_name, index=False)

            _three_group_summary(dist_table, f"EqualStepGroups_{label}")
            _three_group_summary(eqv_table, f"EquiVolumeGroups_{label}")

            # Distribution plots for this variant (folder already created above)
            _plot_composition_from_table(
                dist_table,
                title=f"{sample_name} – Composition vs shell (equal steps, Type A={label})",
                x_label="Distance from surface (µm, equal steps)",
                out_path=dist_plots_dir / f"{sample_name}_EvenSteps_Composition_{label}.png",
                type_a_label=label,
            )
            _plot_smoothed_typeA_curves(
                dist_table,
                title=f"{sample_name} – Smoothed composition (equal steps, Type A={label})",
                x_label="Distance from surface (µm, equal steps)",
                out_path=dist_plots_dir / f"{sample_name}_EvenSteps_Composition_{label}_SmoothedCurves.png",
                type_a_label=label,
            )
            _plot_composition_from_table(
                eqv_table,
                title=f"{sample_name} – Composition vs partition (equal volume, Type A={label})",
                x_label="Distance from surface (µm, equal-volume partitions)",
                out_path=dist_plots_dir / f"{sample_name}_EquiVolume_Composition_{label}.png",
                type_a_label=label,
            )
            _plot_smoothed_typeA_curves(
                eqv_table,
                title=f"{sample_name} – Smoothed composition (equal volume, Type A={label})",
                x_label="Distance from surface (µm, equal-volume partitions)",
                out_path=dist_plots_dir / f"{sample_name}_EquiVolume_Composition_{label}_SmoothedCurves.png",
                type_a_label=label,
            )

    # Simple text summary of overlap for the default config Type A / B
    ch_A = str(cfg.get("cell_type_A_channel", "tumor")).lower()
    ch_B = str(cfg.get("cell_type_B_channel", "fibroblast")).lower()
    default_A = {"nucleus": nucleus_mask, "tumor": tumor_mask, "fibroblast": fibro_mask}.get(ch_A, tumor_mask)
    default_B = {"nucleus": nucleus_mask, "tumor": tumor_mask, "fibroblast": fibro_mask}.get(ch_B, fibro_mask)
    overlap_mask = _ensure_bool(default_A & default_B & convex)
    overlap_voxels = int(np.count_nonzero(overlap_mask))

    summary_path = out_dir / f"{sample_name}_CompositionSummary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Sample: {sample_name}\n")
        f.write(f"Default cell type A channel: {ch_A}\n")
        f.write(f"Default cell type B channel: {ch_B}\n")
        f.write(f"Total overlap voxels (A∩B within convex spheroid, default A/B): {overlap_voxels}\n")

    # Overlap image of convex hull, spheroid mask and fibroblast max projections with 3-group borders
    if shells.size > 0:
        # Max projections
        sph_proj_full = _ensure_bool(sph_mask).max(axis=0)
        fibro_proj = _ensure_bool(fibro_mask).max(axis=0)

        # Keep only the largest connected component of the spheroid projection
        labeled_2d, num_2d = ndi.label(sph_proj_full.astype(bool))
        if num_2d > 0:
            counts_2d = np.bincount(labeled_2d.ravel())
            largest_2d = int(np.argmax(counts_2d[1:]) + 1)
            sph_proj = labeled_2d == largest_2d
        else:
            sph_proj = sph_proj_full

        # 2D convex hull (hole free) of the largest spheroid projection
        convex_proj = convex_hull_image(sph_proj.astype(bool))

        # 2D radial grouping based on convex hull only: distance from hull boundary
        h, w = convex_proj.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        # Distance inside convex hull
        dist2d = ndi.distance_transform_edt(convex_proj.astype(bool))
        maxd2d = float(dist2d.max()) if dist2d.size > 0 else 0.0
        if maxd2d > 0:
            d1 = maxd2d / 3.0
            d2 = 2.0 * maxd2d / 3.0
            # Narrow bands around 1/3 and 2/3 distances
            band1 = (dist2d >= d1 - 0.5) & (dist2d <= d1 + 0.5)
            band2 = (dist2d >= d2 - 0.5) & (dist2d <= d2 + 0.5)
            border1 = ndi.binary_dilation(band1) ^ ndi.binary_erosion(band1)
            border2 = ndi.binary_dilation(band2) ^ ndi.binary_erosion(band2)
        else:
            border1 = np.zeros_like(convex_proj, dtype=bool)
            border2 = np.zeros_like(convex_proj, dtype=bool)

        # Draw base masks in order: spheroid, then fibroblast overwriting spheroid
        # Spheroid max projection: magenta
        rgb[sph_proj, 0] = 1.0
        rgb[sph_proj, 2] = 1.0
        # Fibroblast max projection: yellow, overwriting spheroid where present
        rgb[fibro_proj, :] = [1.0, 1.0, 0.0]

        # Convex hull boundary (outermost): cyan edge, overwriting everything
        convex_edge = ndi.binary_dilation(convex_proj) ^ ndi.binary_erosion(convex_proj)
        rgb[convex_edge, :] = [0.0, 1.0, 1.0]

        # 3-group borders: bright green lines, overwriting everything
        rgb[border1, :] = [0.0, 1.0, 0.0]
        rgb[border2, :] = [0.0, 1.0, 0.0]

        overlap_path = dist_plots_dir / f"{sample_name}_Convex_Spheroid_Fibro_Overlap.png"
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(rgb)
        ax.set_title(f"{sample_name} – convex hull, spheroid, fibroblast and 3-region borders")
        ax.axis("off")

        # Simple legend using colored patches
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#FF00FF", edgecolor="none", label="Spheroid (nucleus-based)"),
            Patch(facecolor="#FFFF00", edgecolor="none", label="Fibroblast"),
            Patch(facecolor="#00FFFF", edgecolor="none", label="Convex hull boundary"),
            Patch(facecolor="#00FF00", edgecolor="none", label="1/3 & 2/3 radial borders"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True)
        plt.tight_layout()
        plt.savefig(str(overlap_path), dpi=200, bbox_inches="tight")
        plt.close(fig)



