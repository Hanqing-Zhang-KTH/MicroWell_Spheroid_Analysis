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

## Required Folder Structure for Input Data

Set `dataset_root` in config/GUI Paths panel to a folder shaped like:

```text
dataset_root/
  <experiment_name>/
    spheroid_1/
      ... 3-channel raw TIFFs + 3-channel manual mask TIFFs ...
    spheroid_2/
      ...
```

Sample discovery uses:

- `data_loading.sample_folder_keyword` (default: `spheroid`)
- suffix `_number` (e.g., `spheroid_1`)
- all 3 channels must be complete (raw + mask) to appear in GUI Samples list.

## GUI Usage Workflow

1. **Paths tab**
   - Set `dataset_root` and `results_root`.
   - Click **Save Paths to JSON**.
2. **Samples tab**
   - Click **Refresh Samples**.
   - Double-click rows to mark `[x]` for selected samples.
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
