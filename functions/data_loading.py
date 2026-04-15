"""
Research group:
Björn Önfelt Group
Department of Applied Physics, Division of Biophysics
Royal Institute of Technology

Coding author:
Hanqing Zhang, Researcher, Royal Institute of Technology, hanzha@kth.se
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tifffile


@dataclass
class SampleDataPaths:
    experiment_name: str
    name: str
    tiff_path: Path


def discover_samples(
    tiff_folder: Path,
    experiment_name: str,
) -> List[SampleDataPaths]:
    """
    Discover samples from a folder of multi-channel TIFF files.

    New input convention (2026-04):
    - Each `.tif/.tiff` file is one sample.
    - File name must contain a suffix at the end starting with '_' (underscore).
      The sample name is that suffix without the underscore.

    Example:
    - `Tiff_17_3_Ch2_A498_BJ_Image1.tif` -> sample name `Image1`

    The TIFF content is expected to contain 6 channels:
    (tumor, fibroblast, nucleus, tumor mask, fibroblast mask, nucleus mask)
    """
    tiff_folder = Path(tiff_folder)
    if not tiff_folder.exists() or not tiff_folder.is_dir():
        return []

    exp = str(experiment_name).strip() or "Default_Experiment"

    def _is_valid_tiff(p: Path) -> bool:
        return p.is_file() and (p.suffix.lower() in {".tif", ".tiff"}) and (not p.name.startswith("."))

    def _sample_name_from_tiff_filename(file_name: str) -> str:
        stem = Path(file_name).stem
        # Require at least one '_' and take the suffix after the last underscore.
        if "_" not in stem:
            raise ValueError(f"Invalid TIFF name (missing '_<SampleName>' suffix): {file_name}")
        suffix = stem.rsplit("_", 1)[-1].strip()
        if not suffix:
            raise ValueError(f"Invalid TIFF name (empty sample suffix): {file_name}")
        return suffix

    collected: dict[Tuple[str, str], SampleDataPaths] = {}
    for p in sorted((x for x in tiff_folder.iterdir() if _is_valid_tiff(x)), key=lambda x: x.name.lower()):
        try:
            sample_name = _sample_name_from_tiff_filename(p.name)
        except ValueError:
            # Skip files that don't follow the naming rule.
            continue
        key = (exp, sample_name)
        if key in collected:
            # Avoid silent overwrite if multiple files map to the same sample name.
            i = 2
            while (exp, f"{sample_name}_{i}") in collected:
                i += 1
            sample_name = f"{sample_name}_{i}"
            key = (exp, sample_name)
        collected[key] = SampleDataPaths(experiment_name=exp, name=sample_name, tiff_path=p)

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


