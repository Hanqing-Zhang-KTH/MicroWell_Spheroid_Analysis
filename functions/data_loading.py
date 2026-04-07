"""
Research group:
Björn Önfelt Group
Department of Applied Physics, Division of Biophysics
Royal Institute of Technology

Coding author:
Hanqing Zhang, Researcher, Royal Institute of Technology, hanzha@kth.se
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import tifffile


@dataclass
class ChannelFiles:
    raw: Path
    mask: Path


@dataclass
class SampleDataPaths:
    experiment_name: str
    name: str
    root: Path
    nucleus: ChannelFiles
    tumor: ChannelFiles
    fibroblast: ChannelFiles


def _is_sample_folder_name(folder_name: str, sample_keyword: str) -> bool:
    """
    Match sample folder names like '<sample_keyword>_<id>', e.g. spheroid_1.
    """
    kw = sample_keyword.strip().lower()
    name = folder_name.strip().lower()
    if not kw:
        return False
    pattern = rf"^{re.escape(kw)}_(\d+)$"
    return re.match(pattern, name) is not None


def _list_experiment_sample_folders(dataset_root: Path, sample_keyword: str) -> List[Tuple[str, Path]]:
    """
    Find sample folders under experiment folders:
      dataset_root/<experiment_name>/<sample_keyword>_<id>
    """
    pairs: List[Tuple[str, Path]] = []
    for exp_dir in sorted(dataset_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        for sample_dir in sorted(exp_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            if _is_sample_folder_name(sample_dir.name, sample_keyword):
                pairs.append((exp_dir.name, sample_dir))
    return pairs


def _sample_name_from_filename(file_name: str, sample_keyword: str) -> str | None:
    """
    Extract sample prefix from file names like:
      spheroid_1_ch0_RFP_HCT_raw.tif  -> spheroid_1
    """
    kw = sample_keyword.strip().lower()
    name = file_name.strip().lower()
    if not kw:
        return None
    m = re.match(rf"^({re.escape(kw)}_\d+)", name)
    if not m:
        return None
    return m.group(1)


def _filter_valid_files(folder: Path) -> List[Path]:
    """Return .tif/.tiff files in folder, skipping hidden files (starting with '.')."""
    files: List[Path] = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if f.name.startswith("."):
            continue
        if f.suffix.lower() in {".tif", ".tiff"}:
            files.append(f)
    return files


def _find_channel_pair(
    files: List[Path],
    name_keyword: str,
    raw_keyword: str,
    mask_keyword: str,
) -> ChannelFiles:
    """Locate raw/mask files for one channel based on name and role keywords."""
    name_kw = name_keyword.lower()
    raw_kw = raw_keyword.lower()
    mask_kw = mask_keyword.lower()

    raw_file = None
    mask_file = None
    for f in files:
        name_lower = f.name.lower()
        if name_kw in name_lower and raw_kw in name_lower:
            raw_file = f
        elif name_kw in name_lower and mask_kw in name_lower:
            mask_file = f

    if raw_file is None or mask_file is None:
        raise FileNotFoundError(
            f"Expected both raw and mask files for channel '{name_keyword}' "
            f"(keywords: raw='{raw_keyword}', mask='{mask_keyword}'). "
            f"Found raw={raw_file}, mask={mask_file}."
        )
    return ChannelFiles(raw=raw_file, mask=mask_file)


def discover_samples(
    dataset_root: Path,
    sample_folder_keyword: str,
    channels_cfg: Dict[str, Dict[str, str]],
) -> List[SampleDataPaths]:
    """
    Discover sample folders and locate required TIFF stacks for each channel.

    Each sample folder must contain exactly three channel pairs:
    - nucleus (e.g. DRAQ5)
    - tumor (e.g. HCT)
    - fibroblast (e.g. BJ)
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists() or not dataset_root.is_dir():
        return []

    nucleus_cfg = channels_cfg["nucleus"]
    tumor_cfg = channels_cfg["tumor"]
    fibro_cfg = channels_cfg["fibroblast"]

    collected: Dict[Tuple[str, str], SampleDataPaths] = {}

    def _try_add_sample(experiment_name: str, sample_name: str, sample_root: Path, files: List[Path]) -> None:
        if not files:
            return
        try:
            nucleus = _find_channel_pair(
                files,
                nucleus_cfg["name_keyword"],
                nucleus_cfg["raw_keyword"],
                nucleus_cfg["mask_keyword"],
            )
            tumor = _find_channel_pair(
                files,
                tumor_cfg["name_keyword"],
                tumor_cfg["raw_keyword"],
                tumor_cfg["mask_keyword"],
            )
            fibro = _find_channel_pair(
                files,
                fibro_cfg["name_keyword"],
                fibro_cfg["raw_keyword"],
                fibro_cfg["mask_keyword"],
            )
        except FileNotFoundError:
            return

        key = (experiment_name, sample_name)
        collected[key] = SampleDataPaths(
            experiment_name=experiment_name,
            name=sample_name,
            root=sample_root,
            nucleus=nucleus,
            tumor=tumor,
            fibroblast=fibro,
        )

    # Layout A: dataset_root/<experiment>/<sample>/files
    for experiment_name, folder in _list_experiment_sample_folders(dataset_root, sample_folder_keyword):
        _try_add_sample(experiment_name, folder.name, folder, _filter_valid_files(folder))

    # Layout B: dataset_root/<experiment>/files named with sample prefix (e.g. spheroid_1_*)
    for exp_dir in sorted(dataset_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_files = _filter_valid_files(exp_dir)
        if not exp_files:
            continue
        grouped: Dict[str, List[Path]] = {}
        for f in exp_files:
            sample_name = _sample_name_from_filename(f.name, sample_folder_keyword)
            if sample_name is None:
                continue
            grouped.setdefault(sample_name, []).append(f)
        for sample_name, files in grouped.items():
            _try_add_sample(exp_dir.name, sample_name, exp_dir, files)

    # Layout C (legacy): dataset_root/<sample>/files (no experiment layer)
    for sample_dir in sorted(dataset_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        if not _is_sample_folder_name(sample_dir.name, sample_folder_keyword):
            continue
        _try_add_sample("default_experiment", sample_dir.name, sample_dir, _filter_valid_files(sample_dir))

    return [collected[k] for k in sorted(collected.keys(), key=lambda x: (x[0].lower(), x[1].lower()))]


def load_tiff_stack(path: Path) -> Tuple[Path, "tifffile.TiffFile", None]:
    """
    Lightweight check for accessibility of a TIFF stack.

    The function mainly serves as a single place to extend I/O checks later
    (e.g. metadata inspection). For now, it just ensures the file can be opened.
    """
    tf = tifffile.TiffFile(str(path))
    tf.close()
    return path, tf, None


