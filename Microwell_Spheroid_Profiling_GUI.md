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

1. **Paths tab**: set `dataset_root` and `results_root`, then save.
2. **Samples tab**: refresh and select valid samples with `[x]`.
3. **Parameters tab**: update non-path settings and save.
4. **Run Profiling**: start run, monitor logs, stop if needed.
5. **Results tab**: validate outputs, run averaging, or create dataset.

## Data Discovery Rule

Expected structure:

`dataset_root/<experiment_name>/<sample_folder_keyword>_<number>`

Example:

- `Dataset/exp_A/spheroid_1`
- `Dataset/exp_A/spheroid_2`
- `Dataset/exp_B/spheroid_1`

Only complete samples are listed as valid.

