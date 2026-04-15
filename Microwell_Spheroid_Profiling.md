# Standalone Microwell Spheroid Profiling (Windows + CUDA)

Research group: Björn Önfelt Group, KTH  
Coding author: Hanqing Zhang (hanzha@kth.se)

## What This Folder Contains

- `Microwell_Spheroid_Profiling_GUI.py`: GUI entry point.
- `Spheroid_Profiling.py`: profiling pipeline used by GUI.
- `Averaging_Pattern.py`: averaging/alignment post-analysis.
- `functions/`: processing modules required by pipeline.
- `Networks/`: local network definitions and model files.
- `Config/spheroid_config.json`: runtime configuration.
- `requirements_spheroid_gui.txt`: dependencies for private venv.

## First Run on Windows

1. Open this folder.
2. Double-click `Run_GUI_Windows.bat` (or run `python Microwell_Spheroid_Profiling_GUI.py`).
3. On first launch, GUI auto-creates `.spheroid_gui_venv` locally in this folder.
4. Dependencies are installed into this private venv only (system Python unchanged).
5. If NVIDIA CUDA is detected, GUI installs CUDA PyTorch wheels automatically.

## Input Data (New Convention)

Input is now a **folder of TIFF files** (absolute path is fine), e.g. `D:\Data_260415`.

Each `.tif/.tiff` file is treated as one sample. The filename must end with a suffix starting with `_`
(before the extension). The sample name is that suffix without `_`.

Example:

- `Tiff_17_3_Ch2_A498_BJ_Image1.tif` → sample name: `Image1`

Each TIFF is expected to have shape `(6, Z, Y, X)` with channels:

1. tumor raw
2. fibroblast raw
3. nucleus raw
4. tumor mask
5. fibroblast mask
6. nucleus mask

## GUI Usage Workflow

1. **Samples tab**
   - Click **Select Folder** to choose the folder containing TIFF files.
   - Enter an experiment name (default `Default_Experiment`) and click **Confirm**.
   - Set **Results root** if needed and click **Confirm**.
   - Click **Refresh Samples** and double-click rows to mark `[x]` for selected samples.
3. **Parameters tab**
   - Edit non-path settings (quantification/post-analysis/etc.).
   - Click **Save Changes**.
4. **Run Profiling**
   - Click **Run Profiling** (button auto-disables while running).
   - Use **Stop** to terminate current processing immediately.
5. **Results tab**
   - Refresh results and validate outputs.
   - Select result samples for **Averaging Pattern** or **Create Dataset**.

## Result Structure

Outputs are saved as:

```text
results_root/
  <experiment_name>/
    <sample_name>/
      Input/
      Thresholding/
      DL_masks/
      Refinement/
      ImageMask*/
      ... profiling statistics/plots ...
```

Averaging outputs are saved under:

- `results_root/Averaging_YYYYMMDD_HHMMSS/`

## Config Location and Recommended Settings

Config file:

- `Config/spheroid_config.json`

Recommended baseline:

- `quantification.mode = "mix_mode"`
- In `quantification.mix_mode`: use per-channel `global`, `3d`, `2d`, or `manual`
- For manual masks: non-zero values are treated as object (`>0`)
- Keep `post_analysis.watershed_downsize_ratio = 1.0` for full quality
- Set per-channel cell separation control in `post_analysis.apply_cell_separation`
- Set averaging cell separation control in `post_analysis.averaging.apply_cell_separation`

## CUDA / Network Notes

- Keep trained model files in `Networks/Cells3D` and/or `Networks/Cells2D`.
- If models are missing, DL quantification will fail; use `global` or `manual` until models are added.
- This standalone is intended for Windows CUDA machines with internet access on first run.

## Rebuild Standalone After Code Changes

From project root, run:

```bash
python build_standalone_bundle.py
```

This refreshes the standalone folder with your latest code.
