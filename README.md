# Microwell Spheroid Profiling

Windows-focused GUI pipeline for spheroid quantification, post-analysis, and composition profiling.

Use `Microwell_Spheroid_Profiling_GUI.py` as the primary entry point.

## Quick Start

1. Run `Run_GUI_Windows.bat` (double-click) or:
   - `python Microwell_Spheroid_Profiling_GUI.py`
2. In the GUI Paths tab, set `dataset_root` and `results_root`.
3. Refresh samples, select `[x]`, and click **Run Profiling**.

Detailed usage is documented in:

- `Microwell_Spheroid_Profiling.md`
- `Microwell_Spheroid_Profiling_GUI.md`

## Repository Notes

- This project may include `.pth` model weights under `Networks/`.
- `.pth` files over 100 MB require Git LFS on GitHub.
