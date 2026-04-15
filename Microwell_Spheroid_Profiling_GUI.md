# Microwell Spheroid Profiling GUI

Research group: Björn Önfelt Group, Department of Applied Physics, Division of Biophysics, Royal Institute of Technology  
Coding author: Hanqing Zhang, Researcher, Royal Institute of Technology, [hanzha@kth.se](mailto:hanzha@kth.se)

This GUI is a config-driven front-end for `Spheroid_Profiling.py` and `Config/spheroid_config.json`.

## Start

Run from the project root:

```bash
python Microwell_Spheroid_Profiling_GUI.py
```

`tkinter` is used, so it works on Windows by default and is also cross-platform on macOS/Linux when Python `tk` support is installed.

## Isolated Dependency Bootstrap

The GUI auto-runs in a private project virtual environment:

- venv path: `.spheroid_gui_venv`
- dependency file: `requirements_spheroid_gui.txt`
- no modification to system Python or user global environment

On first start (or when requirements change), the app automatically:

1. Creates `.spheroid_gui_venv`
2. Installs dependencies in that venv
3. Installs PyTorch stack automatically:
   - NVIDIA detected: CUDA wheel (`cu121`)
   - no NVIDIA detected: CPU wheel
4. Relaunches itself inside that venv

## Main Workflow

1. **Paths tab**: set `results_root` (default `Results`). `dataset_root` can stay as default and is only used as a convenient default folder in the GUI.
2. **Samples tab**:
   - Select the **sample folder** containing your `.tif/.tiff` files (absolute path is fine, e.g. `D:\Data_260415`).
   - Enter an **experiment name** and click **Confirm** (default: `Default_Experiment`).
   - Click **Refresh Samples** and select samples with `[x]`.
3. **Parameters tab**: update non-path settings and save.
4. **Run Profiling**: start run, monitor logs, stop if needed.
5. **Results tab**: validate outputs, run averaging, or create dataset.

## Data Discovery Rule (New Input Convention)

Expected structure is now a **single folder of TIFF files** (no `Dataset/<experiment>/<sample>` layout required):

```text
<your_sample_folder>/
  *.tif / *.tiff
```

Each TIFF file must end with a suffix starting with `_` in the filename (before `.tif/.tiff`).
The **sample name** is the substring after the final underscore.

Example:

- `Tiff_17_3_Ch2_A498_BJ_Image1.tif` → sample name: `Image1`

Each TIFF is expected to contain **6 channels** with shape `(6, Z, Y, X)`:

1. tumor (raw intensity)
2. fibroblast (raw intensity)
3. nucleus (raw intensity)
4. tumor mask
5. fibroblast mask
6. nucleus mask

Only complete samples are listed as valid.

