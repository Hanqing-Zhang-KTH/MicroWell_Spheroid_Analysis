# Microwell Spheroid Profiling

GUI pipeline for spheroid quantification, post-analysis, and composition profiling. Tested in Windows.

Use `Microwell_Spheroid_Profiling_GUI.py` as the primary entry point.

## Quick Start

1. Run `Run_GUI_Windows.bat` (double-click) or:
   - `python Microwell_Spheroid_Profiling_GUI.py`
2. In the GUI Samples tab, select the TIFF folder and set experiment name.
3. Refresh samples, select `[x]`, and click **Run Profiling**.

## Release / Download (For Non-Technical Users)

If you just want to run the software:

1. Open the GitHub repository page.
2. Click the green **Code** button.
3. Click **Download ZIP**.
4. Extract the ZIP to a local folder (for example `D:\Microwell_Spheroid_Profiling`).
5. Open that folder and double-click `Run_GUI_Windows.bat`.
6. On first run, wait for automatic dependency setup to finish.

If Windows shows a security warning, choose **More info** -> **Run anyway** (only if you trust this source).

Detailed usage is documented in:

- `Microwell_Spheroid_Profiling.md`
- `Microwell_Spheroid_Profiling_GUI.md`

## Repository Notes

- This project may include `.pth` model weights under `Networks/`.
- `.pth` files over 100 MB require Git LFS on GitHub.

## License

- License: GNU General Public License v3.0 (`LICENSE`)
- Copyright and attribution details: `COPYRIGHT.md`

## Project Governance

- Contribution process and branch workflow: `CONTRIBUTING.md`
- Owner-review rule for all changes: `.github/CODEOWNERS`
- Pull request checklist: `.github/pull_request_template.md`
- One-time repository owner setup (visibility, branch protection, collaborators): `MAINTAINER_SETUP.md`
