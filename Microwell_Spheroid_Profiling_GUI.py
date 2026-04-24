"""
Microwell Spheroid Profiling GUI for config-driven processing.

Research group:
Björn Önfelt Group
Department of Applied Physics, Division of Biophysics
Royal Institute of Technology

Coding author:
Hanqing Zhang, Researcher, Royal Institute of Technology, hanzha@kth.se
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import venv
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as np
    import tifffile
    from functions.data_loading import SampleDataPaths, discover_samples
    from functions.io_utils import load_json_config
except Exception:
    # Allow bootstrap to run even when scientific stack is not yet installed
    # in the system interpreter. These modules are loaded after venv bootstrap.
    np = None  # type: ignore[assignment]
    tifffile = None  # type: ignore[assignment]
    SampleDataPaths = Any  # type: ignore[assignment]
    discover_samples = None  # type: ignore[assignment]
    load_json_config = None  # type: ignore[assignment]


def _ensure_runtime_imports() -> None:
    """Import runtime modules after private-venv bootstrap."""
    global np, tifffile, SampleDataPaths, discover_samples, load_json_config
    if np is not None and tifffile is not None and discover_samples is not None and load_json_config is not None:
        return
    import numpy as _np
    import tifffile as _tifffile
    from functions.data_loading import SampleDataPaths as _SampleDataPaths, discover_samples as _discover_samples
    from functions.io_utils import load_json_config as _load_json_config

    np = _np
    tifffile = _tifffile
    SampleDataPaths = _SampleDataPaths
    discover_samples = _discover_samples
    load_json_config = _load_json_config


def _read_tiff_stack_robust(path: Path) -> np.ndarray:
    """
    Read TIFF robustly while avoiding avoidable RAM spikes.
    - Prefer tifffile.imread when it already returns 3D+
    - For multi-page 2D TIFFs, materialize as (Z, Y, X) using series/pages
    """
    try:
        arr = np.asarray(tifffile.imread(str(path)))
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF: {path} ({exc})") from exc
    if arr.ndim >= 3:
        return arr
    with tifffile.TiffFile(str(path)) as tf:
        if tf.series:
            try:
                series_arr = np.asarray(tf.series[0].asarray())
                if series_arr.ndim >= 3:
                    return series_arr
            except Exception:
                pass
        if len(tf.pages) > 1:
            first = tf.pages[0].asarray()
            first_shape = first.shape
            for p in tf.pages[1:]:
                if p.shape != first_shape:
                    return arr
            out = np.empty((len(tf.pages),) + first_shape, dtype=first.dtype)
            out[0] = first
            for i, p in enumerate(tf.pages[1:], start=1):
                out[i] = p.asarray()
            return out
    return arr


def _venv_python_path(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _has_nvidia_gpu() -> bool:
    """Best-effort NVIDIA detection without importing torch."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return completed.returncode == 0 and "GPU" in completed.stdout
    except FileNotFoundError:
        return False


def _torch_cuda_available(py_exe: Path) -> bool:
    """Check whether torch in target interpreter can see CUDA."""
    check_code = (
        "import torch; "
        "print('1' if torch.cuda.is_available() else '0')"
    )
    completed = subprocess.run(
        [str(py_exe), "-c", check_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip().endswith("1")


def _venv_health_check(py_exe: Path) -> tuple[bool, str]:
    """
    Validate that core scientific stack and DL stack can be imported together.
    Returns (ok, message).
    """
    check_code = (
        "import numpy, scipy, skimage, matplotlib, pandas, tifffile, torch, torchvision, monai; "
        "print('OK')"
    )
    completed = subprocess.run(
        [str(py_exe), "-c", check_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and "OK" in completed.stdout:
        return True, "ok"
    msg = completed.stderr.strip() or completed.stdout.strip() or f"exit={completed.returncode}"
    return False, msg


def _needs_core_pin_fix(py_exe: Path) -> bool:
    """
    Detect whether venv has versions known to conflict with this pipeline.
    Returns True when numpy>=2 or setuptools>=82.
    """
    check_code = (
        "from importlib.metadata import version; "
        "def major(v):\n"
        "    return int(v.split('.')[0]);\n"
        "n = version('numpy'); s = version('setuptools'); "
        "print('1' if (major(n) >= 2 or major(s) >= 82) else '0')"
    )
    completed = subprocess.run(
        [str(py_exe), "-c", check_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return completed.returncode != 0 or completed.stdout.strip().endswith("1")


def _install_torch_stack(py_exe: Path, project_root: Path, want_cuda: bool) -> None:
    variant = "cu121" if want_cuda else "cpu"
    index_url = f"https://download.pytorch.org/whl/{variant}"
    print(f"[bootstrap] Installing PyTorch stack ({variant}) in private venv...")
    _run_cmd(
        [
            str(py_exe),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "--index-url",
            index_url,
            "--extra-index-url",
            "https://pypi.org/simple",
        ],
        project_root,
    )
    _run_cmd([str(py_exe), "-m", "pip", "install", "--upgrade", "monai==1.5.2"], project_root)


def _repair_binary_stack(py_exe: Path, project_root: Path) -> None:
    """
    Repair ABI-related conflicts quickly without touching system environment.
    Only runs in the private venv.
    """
    print("[bootstrap] Repairing binary compatibility in private venv...")
    _run_cmd([str(py_exe), "-m", "pip", "install", "--upgrade", "numpy<2", "setuptools<82"], project_root)
    _run_cmd(
        [
            str(py_exe),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "scipy",
            "scikit-image",
            "matplotlib",
            "pandas",
            "tifffile",
        ],
        project_root,
    )


def _python_minor(py_exe: Path) -> tuple[int, int]:
    completed = subprocess.run(
        [str(py_exe), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Cannot determine Python version for {py_exe}: {completed.stderr.strip()}")
    parts = completed.stdout.strip().split(".")
    if len(parts) < 2:
        raise RuntimeError(f"Unexpected Python version output: '{completed.stdout.strip()}'")
    return int(parts[0]), int(parts[1])


def _assert_supported_python_for_torch(py_exe: Path) -> None:
    major, minor = _python_minor(py_exe)
    if major != 3 or minor < 10 or minor > 12:
        raise RuntimeError(
            "Unsupported Python version for bundled torch stack: "
            f"{major}.{minor}. Please run with Python 3.10, 3.11, or 3.12 "
            "(3.11 recommended)."
        )


def ensure_isolated_runtime(project_root: Path) -> None:
    """
    Ensure GUI runs in a private project venv so user/system Python is untouched.
    First run will auto-create and install dependencies from requirements file.
    """
    venv_dir = project_root / ".spheroid_gui_venv"
    req_path = project_root / "requirements_spheroid_gui.txt"
    marker = venv_dir / ".deps_ready"
    req_hash_marker = venv_dir / ".requirements_hash"
    torch_marker = venv_dir / ".torch_variant"
    target_py = _venv_python_path(venv_dir)
    current_py = Path(sys.executable).resolve()

    if not req_path.exists():
        raise FileNotFoundError(f"Missing dependency file: {req_path}")

    if not target_py.exists():
        print(f"[bootstrap] Creating private venv: {venv_dir}")
        builder = venv.EnvBuilder(with_pip=True, clear=False)
        builder.create(str(venv_dir))

    _assert_supported_python_for_torch(target_py)

    expected_hash = _sha256_file(req_path)
    current_hash = req_hash_marker.read_text(encoding="utf-8").strip() if req_hash_marker.exists() else ""
    deps_need_install = (not marker.exists()) or (current_hash != expected_hash)
    if deps_need_install:
        print("[bootstrap] Installing/updating dependencies in private venv...")
        _run_cmd([str(target_py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<82"], project_root)
        _run_cmd([str(target_py), "-m", "pip", "install", "-r", str(req_path)], project_root)
        req_hash_marker.write_text(expected_hash + "\n", encoding="utf-8")

    # Always enforce GPU-first torch variant inside private venv.
    has_nvidia = _has_nvidia_gpu()
    desired_variant = "cu121" if has_nvidia else "cpu"
    current_variant = torch_marker.read_text(encoding="utf-8").strip() if torch_marker.exists() else ""
    core_pin_fix_needed = _needs_core_pin_fix(target_py)
    if core_pin_fix_needed:
        print("[bootstrap] Repairing pinned core packages (numpy<2, setuptools<82)...")
        _run_cmd([str(target_py), "-m", "pip", "install", "--upgrade", "numpy<2", "setuptools<82"], project_root)

    torch_needs_install = deps_need_install or (current_variant != desired_variant)
    if has_nvidia and not _torch_cuda_available(target_py):
        torch_needs_install = True

    if torch_needs_install:
        _install_torch_stack(target_py, project_root, want_cuda=has_nvidia)
        torch_marker.write_text(desired_variant + "\n", encoding="utf-8")

    # Final consistency pass: repair only if imports still fail.
    healthy, reason = _venv_health_check(target_py)
    if not healthy:
        print(f"[bootstrap] Health check failed: {reason}")
        _repair_binary_stack(target_py, project_root)
        healthy2, reason2 = _venv_health_check(target_py)
        if not healthy2:
            raise RuntimeError(f"Private venv health check failed after repair: {reason2}")

    marker.write_text("ok\n", encoding="utf-8")
    print("[bootstrap] Dependencies ready.")

    target_py_resolved = target_py.resolve()
    if current_py != target_py_resolved:
        print("[bootstrap] Relaunching GUI in private venv...")
        env = os.environ.copy()
        env["SPHEROID_GUI_ISOLATED"] = "1"
        code = subprocess.call([str(target_py_resolved), str(Path(__file__).resolve())], cwd=str(project_root), env=env)
        sys.exit(code)


class SpheroidProfilingGUI(tk.Tk):
    def __init__(self, project_root: Path, config_path: Path) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.config_path = Path(config_path)
        self.title("Microwell Spheroid Profiling GUI")
        self.geometry("1000x680")
        self.minsize(860, 560)

        self.cfg: dict = {}
        self.sample_selected: dict[str, bool] = {}
        self.sample_map: dict[str, SampleDataPaths] = {}
        self.result_selected: dict[str, bool] = {}
        self.result_sample_paths: dict[str, Path] = {}
        self.result_validation_ok: dict[str, bool] = {}
        self._running = False
        self._stop_requested = False
        self._active_proc: subprocess.Popen[str] | None = None
        self._avg_proc: subprocess.Popen[str] | None = None
        self._averaging_running = False
        self._in_error_trace = False
        self._suppress_warning_context = False
        self._msg_queue: queue.Queue[str] = queue.Queue()
        self.input_folder_var = tk.StringVar(value=str((self.project_root / "Dataset").resolve()))
        self.results_root_var = tk.StringVar(value=str((self.project_root / "Results").resolve()))
        self.experiment_entry_var = tk.StringVar(value="Default_Experiment")
        self.active_experiment_name: str = "Default_Experiment"

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_config_into_ui()
        self.refresh_samples(silent=True)
        self.refresh_results_tree()
        self.after(400, self._drain_messages)
        self.after(5000, self._auto_refresh)

    def _on_close(self) -> None:
        """
        Ensure any spawned worker subprocesses are terminated when GUI closes.
        This prevents orphaned averaging/profiling runs continuing in background.
        """
        try:
            self._stop_requested = True
            for proc in [self._active_proc, self._avg_proc]:
                if proc is not None and proc.poll() is None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
        finally:
            try:
                self.destroy()
            except Exception:
                pass

    @staticmethod
    def _sanitize_paths(paths_cfg: dict) -> dict:
        """Drop legacy *_ref_root entries; pipeline is self-contained now."""
        cleaned: dict[str, str] = {}
        for k, v in paths_cfg.items():
            if str(k).lower().endswith("_ref_root"):
                continue
            cleaned[str(k)] = v
        return cleaned

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text=f"Config: {self.config_path}").pack(side="left")
        ttk.Button(top, text="Reload JSON", command=self._reload_json).pack(side="right", padx=(8, 0))
        self.stop_btn = ttk.Button(top, text="Stop", command=self._stop_processing, state="disabled")
        self.stop_btn.pack(side="right", padx=(8, 0))
        self.run_btn = ttk.Button(top, text="Run Profiling", command=self._run_selected_samples)
        self.run_btn.pack(side="right", padx=(8, 0))

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=4)

        self.samples_tab = ttk.Frame(notebook, padding=8)
        self.results_tab = ttk.Frame(notebook, padding=8)
        self.params_tab = ttk.Frame(notebook, padding=8)
        notebook.add(self.samples_tab, text="Samples")
        notebook.add(self.results_tab, text="Results")
        notebook.add(self.params_tab, text="Parameters")

        self._build_samples_tab()
        self._build_results_tab()
        self._build_params_tab()

        log_wrap = ttk.LabelFrame(self, text="Run Log", padding=8)
        log_wrap.pack(fill="both", expand=False, padx=8, pady=(0, 8))
        self.log_text = tk.Text(log_wrap, height=6, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

        bottom = ttk.Frame(self, padding=(8, 0, 8, 8))
        bottom.pack(fill="x")
        copyright_text = "Björn Önfelt Group, KTH | Author: Hanqing Zhang (hanzha@kth.se)"
        ttk.Label(bottom, text=copyright_text, foreground="#555555").pack(side="right")

    def _build_samples_tab(self) -> None:
        row1 = ttk.Frame(self.samples_tab)
        row1.pack(fill="x", pady=(0, 8))
        ttk.Label(row1, text="Sample folder:").pack(side="left")
        ttk.Entry(row1, textvariable=self.input_folder_var, width=58).pack(side="left", padx=(6, 6))
        ttk.Button(row1, text="Select Folder", command=self._choose_input_folder).pack(side="left")

        ttk.Label(row1, text="Experiment name:").pack(side="left", padx=(12, 4))
        ttk.Entry(row1, textvariable=self.experiment_entry_var, width=22).pack(side="left")
        ttk.Button(row1, text="Confirm", command=self._confirm_experiment_name).pack(side="left", padx=(6, 10))

        ttk.Button(row1, text="Refresh Samples", command=self.refresh_samples).pack(side="left")
        self.samples_info_label = ttk.Label(row1, text="")
        self.samples_info_label.pack(side="left", padx=(12, 0))

        row2 = ttk.Frame(self.samples_tab)
        row2.pack(fill="x", pady=(0, 8))
        ttk.Label(row2, text="Results root:").pack(side="left")
        ttk.Entry(row2, textvariable=self.results_root_var, width=74).pack(side="left", padx=(6, 6))
        ttk.Button(row2, text="Confirm", command=self._confirm_results_root).pack(side="left")
        ttk.Button(row2, text="Reset to default", command=self._reset_to_default).pack(side="left", padx=(8, 0))

        cols = ("selected", "experiment", "sample", "status")
        self.samples_tree = ttk.Treeview(self.samples_tab, columns=cols, show="headings", height=14)
        self.samples_tree.heading("selected", text="Run")
        self.samples_tree.heading("experiment", text="Experiment")
        self.samples_tree.heading("sample", text="Sample")
        self.samples_tree.heading("status", text="Validation")
        self.samples_tree.column("selected", width=60, anchor="center")
        self.samples_tree.column("experiment", width=260, anchor="w")
        self.samples_tree.column("sample", width=260, anchor="w")
        self.samples_tree.column("status", width=220, anchor="w")
        self.samples_tree.pack(fill="both", expand=True)
        self.samples_tree.bind("<Double-1>", self._toggle_sample_row)

        help_lbl = ttk.Label(
            self.samples_tab,
            text="Double-click a row to toggle Run [x]/[ ]. Samples are TIFF files in the selected folder.",
            foreground="#555555",
        )
        help_lbl.pack(anchor="w", pady=(6, 0))

    def _choose_input_folder(self) -> None:
        initial = self.input_folder_var.get().strip()
        init_dir = initial if initial and Path(initial).exists() else str(self.project_root.resolve())
        chosen = filedialog.askdirectory(initialdir=init_dir, title="Select folder containing TIFF files")
        if chosen:
            self.input_folder_var.set(chosen)
            self.refresh_samples()

    def _confirm_experiment_name(self) -> None:
        raw = self.experiment_entry_var.get().strip()
        name = raw or "Default_Experiment"
        name = re.sub(r"[<>:\"/\\\\|?*]+", "_", name).strip()
        if not name:
            name = "Default_Experiment"
        self.active_experiment_name = name
        self.experiment_entry_var.set(name)
        self._log(f"Experiment name set to: {name}")
        self.refresh_samples(silent=True)

    def _confirm_results_root(self) -> None:
        raw = self.results_root_var.get().strip()
        if not raw:
            messagebox.showerror("Results root", "Results root cannot be empty.")
            return
        p = Path(raw)
        if not p.is_absolute():
            p = (self.project_root / p).resolve()
        self.results_root_var.set(str(p))
        self.cfg.setdefault("paths", {})
        self.cfg["paths"]["results_root"] = str(p)
        self._write_config()
        self._log(f"Results root confirmed: {p}")
        self.refresh_results_tree()

    def _reset_to_default(self) -> None:
        default_sample = (self.project_root / "Dataset").resolve()
        default_results = (self.project_root / "Results").resolve()
        self.input_folder_var.set(str(default_sample))
        self.results_root_var.set(str(default_results))
        self.active_experiment_name = "Default_Experiment"
        self.experiment_entry_var.set("Default_Experiment")
        self.cfg.setdefault("paths", {})
        self.cfg["paths"]["results_root"] = str(default_results)
        self._write_config()
        self._log("Reset to default: sample folder, results root, and experiment name.")
        self.refresh_samples(silent=True)
        self.refresh_results_tree()

    def _build_results_tab(self) -> None:
        row = ttk.Frame(self.results_tab)
        row.pack(fill="x", pady=(0, 8))
        ttk.Button(row, text="Refresh Results", command=self.refresh_results_tree).pack(side="left")
        ttk.Button(row, text="Open Selected Folder", command=self._open_selected_result_folder).pack(side="left", padx=(8, 0))
        self.avg_btn = ttk.Button(row, text="Averaging Pattern", command=self._run_averaging_pattern)
        self.avg_btn.pack(side="left", padx=(8, 0))
        ttk.Label(row, text="Channel:").pack(side="left", padx=(12, 4))
        self.dataset_channel_var = tk.StringVar(value="nucleus")
        self.dataset_channel_box = ttk.Combobox(
            row,
            textvariable=self.dataset_channel_var,
            values=["nucleus", "tumor", "fibroblast"],
            width=10,
            state="readonly",
        )
        self.dataset_channel_box.pack(side="left")
        self.create_dataset_btn = ttk.Button(row, text="Create Dataset", command=self._create_dataset_from_results)
        self.create_dataset_btn.pack(side="left", padx=(8, 0))

        self.results_root_label = ttk.Label(row, text="")
        self.results_root_label.pack(side="left", padx=(12, 0))

        self.results_tree = ttk.Treeview(
            self.results_tab,
            columns=("selected", "experiment", "sample", "validation", "path"),
            show="headings",
            height=14,
        )
        self.results_tree.heading("selected", text="Use")
        self.results_tree.heading("experiment", text="Experiment")
        self.results_tree.heading("sample", text="Sample")
        self.results_tree.heading("validation", text="Validation")
        self.results_tree.heading("path", text="Path")
        self.results_tree.column("selected", width=55, anchor="center")
        self.results_tree.column("experiment", width=220, anchor="w")
        self.results_tree.column("sample", width=220, anchor="w")
        self.results_tree.column("validation", width=220, anchor="w")
        self.results_tree.column("path", width=260, anchor="w")
        self.results_tree.pack(fill="both", expand=True)
        self.results_tree.bind("<Double-1>", self._toggle_result_row)

    def _build_params_tab(self) -> None:
        row = ttk.Frame(self.params_tab)
        row.pack(fill="x", pady=(0, 8))
        ttk.Button(row, text="Save Changes", command=self._save_parameters_to_json).pack(side="left")
        ttk.Label(
            row,
            text="Edit all non-path settings (data_loading, quantification, preprocessing, post_analysis, etc.).",
            foreground="#555555",
        ).pack(side="left", padx=(10, 0))

        self.params_text = tk.Text(self.params_tab, wrap="none")
        self.params_text.pack(fill="both", expand=True)

    def _load_config_into_ui(self) -> None:
        self.cfg = load_json_config(self.config_path)
        self.cfg["paths"] = self._sanitize_paths(self.cfg.get("paths", {}))
        rr = str(self.cfg.get("paths", {}).get("results_root", "")).strip()
        if rr:
            p = Path(rr)
            if not p.is_absolute():
                p = (self.project_root / p).resolve()
            self.results_root_var.set(str(p))
        self._render_params_text()

    def _render_params_text(self) -> None:
        non_paths = deepcopy(self.cfg)
        non_paths.pop("paths", None)
        self.params_text.delete("1.0", "end")
        self.params_text.insert("1.0", json.dumps(non_paths, indent=2))

    def _reload_json(self) -> None:
        self._load_config_into_ui()
        self.refresh_samples()
        self.refresh_results_tree()

    def _save_parameters_to_json(self) -> None:
        raw = self.params_text.get("1.0", "end").strip()
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            messagebox.showerror("Invalid JSON", f"Parameters JSON parse failed:\n{exc}")
            return
        if not isinstance(parsed, dict):
            messagebox.showerror("Invalid JSON", "Parameters content must be a JSON object.")
            return

        self.cfg.setdefault("paths", {})
        self.cfg["paths"]["results_root"] = self.results_root_var.get().strip() or str((self.project_root / "Results").resolve())
        self.cfg = {"paths": self.cfg["paths"], **parsed}
        self._write_config()
        self.refresh_samples()
        self.refresh_results_tree()

    def _write_config(self) -> None:
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2)

    def _build_runtime_cfg(self) -> dict:
        # Runtime uses the current in-memory config plus current path fields.
        cfg = deepcopy(self.cfg)
        cfg.setdefault("paths", {})
        cfg["paths"]["results_root"] = self.results_root_var.get().strip() or str((self.project_root / "Results").resolve())
        return cfg

    def refresh_samples(self, silent: bool = False) -> None:
        try:
            folder = Path(self.input_folder_var.get().strip() or (self.project_root / "Dataset"))
            exp = self.active_experiment_name or (self.experiment_entry_var.get().strip() or "Default_Experiment")
            samples = discover_samples(tiff_folder=folder, experiment_name=exp)
        except Exception as exc:
            if not silent:
                messagebox.showerror("Sample Discovery Error", str(exc))
            self.samples_info_label.config(text=f"Discovery failed: {exc}")
            return

        prev_selected = dict(self.sample_selected)
        self.sample_map.clear()
        self.samples_tree.delete(*self.samples_tree.get_children())

        for s in samples:
            sample_id = f"{s.experiment_name}/{s.name}"
            selected = prev_selected.get(sample_id, False)
            self.sample_selected[sample_id] = selected
            self.sample_map[sample_id] = s
            mark = "[x]" if selected else "[ ]"
            self.samples_tree.insert(
                "",
                "end",
                iid=sample_id,
                values=(mark, s.experiment_name, s.name, "6-channel tiff ready"),
            )

        self.samples_info_label.config(
            text=f"Input folder: {folder} | Experiment: {self.active_experiment_name} | Samples: {len(samples)}"
        )

    def _should_show_log_line(self, text: str) -> bool:
        s = text.strip()
        if not s:
            return False

        # Keep full traceback/error blocks visible.
        if "Traceback (most recent call last):" in s:
            self._in_error_trace = True
            return True
        if self._in_error_trace:
            if s.startswith("Run failed") or s.startswith("[") or s.startswith("All selected samples"):
                self._in_error_trace = False
            return True

        # Hide known warning lines and their immediate source-context line.
        if "FutureWarning:" in s or "UserWarning:" in s or "DeprecationWarning:" in s:
            self._suppress_warning_context = True
            return False
        if self._suppress_warning_context:
            if text.startswith("  ") or text.startswith("\t"):
                self._suppress_warning_context = False
                return False
            self._suppress_warning_context = False

        # Hide redundant per-subprocess discovery lines.
        if s.startswith("Found ") and "spheroid samples." in s:
            return False
        if "Processing sample:" in s and s.startswith("["):
            return False

        return True

    def _toggle_sample_row(self, _event: tk.Event) -> None:
        selected_iid = self.samples_tree.focus()
        if not selected_iid:
            return
        current = bool(self.sample_selected.get(selected_iid, False))
        self.sample_selected[selected_iid] = not current
        vals = list(self.samples_tree.item(selected_iid, "values"))
        vals[0] = "[x]" if not current else "[ ]"
        self.samples_tree.item(selected_iid, values=vals)

    def refresh_results_tree(self) -> None:
        self.results_tree.delete(*self.results_tree.get_children())
        cfg = self._build_runtime_cfg()
        rr = str(cfg.get("paths", {}).get("results_root", "Results")).strip() or "Results"
        rp = Path(rr)
        results_root = rp if rp.is_absolute() else (self.project_root / rp)
        results_root = results_root.resolve()
        self.results_root_label.config(text=f"Results root: {results_root}")
        if not results_root.exists():
            return

        legacy_markers = {"ImageMask", "ImageMaskDL", "ImageMaskMix", "Thresholding", "DL_masks", "Refinement"}
        prev_selected = dict(self.result_selected)
        self.result_selected.clear()
        self.result_sample_paths.clear()
        self.result_validation_ok.clear()

        for first_dir in sorted(results_root.iterdir()):
            if not first_dir.is_dir():
                continue
            subdirs = [d.name for d in first_dir.iterdir() if d.is_dir()]
            is_legacy_sample = any(name in legacy_markers for name in subdirs)

            if is_legacy_sample:
                sid = f"legacy::{first_dir}"
                ok, status = self._validate_result_sample(first_dir)
                sel = prev_selected.get(sid, False)
                self.result_selected[sid] = sel
                self.result_sample_paths[sid] = first_dir
                self.result_validation_ok[sid] = ok
                self.results_tree.insert(
                    "",
                    "end",
                    iid=sid,
                    values=("[x]" if sel else "[ ]", "legacy", first_dir.name, status, str(first_dir)),
                )
                continue

            for sample_dir in sorted(first_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                sid = f"sample::{sample_dir}"
                ok, status = self._validate_result_sample(sample_dir)
                sel = prev_selected.get(sid, False)
                self.result_selected[sid] = sel
                self.result_sample_paths[sid] = sample_dir
                self.result_validation_ok[sid] = ok
                self.results_tree.insert(
                    "",
                    "end",
                    iid=sid,
                    values=("[x]" if sel else "[ ]", first_dir.name, sample_dir.name, status, str(sample_dir)),
                )

    def _validate_result_sample(self, sample_dir: Path) -> tuple[bool, str]:
        sample_name = sample_dir.name
        has_tiff = any(sample_dir.rglob("*.tif")) or any(sample_dir.rglob("*.tiff"))
        has_stats = (sample_dir / f"{sample_name}_CompositionStat.xlsx").exists()
        has_summary = (sample_dir / f"{sample_name}_CompositionSummary.txt").exists()
        has_profile_png = (
            any((sample_dir / "DistributionPlots").glob("*.png"))
            or any((sample_dir / "DistPlots").glob("*.png"))
            or any(sample_dir.glob("*Composition*.png"))
        )

        missing: list[str] = []
        if not has_tiff:
            missing.append("tiff")
        if not has_stats:
            missing.append("stats")
        if not has_summary:
            missing.append("summary")
        if not has_profile_png:
            missing.append("plots")
        if missing:
            return False, "Missing: " + ", ".join(missing)
        return True, "Ready"

    def _toggle_result_row(self, _event: tk.Event) -> None:
        selected_iid = self.results_tree.focus()
        if not selected_iid:
            return
        if selected_iid not in self.result_sample_paths:
            return
        vals = list(self.results_tree.item(selected_iid, "values"))
        current = bool(self.result_selected.get(selected_iid, False))
        self.result_selected[selected_iid] = not current
        vals[0] = "[x]" if not current else "[ ]"
        self.results_tree.item(selected_iid, values=vals)

    def _open_selected_result_folder(self) -> None:
        node = self.results_tree.focus()
        if not node:
            messagebox.showinfo("Open Folder", "Select an experiment or sample folder in Results first.")
            return
        vals = self.results_tree.item(node, "values")
        if len(vals) < 5:
            return
        folder = Path(vals[4])
        if not folder.exists():
            messagebox.showerror("Open Folder", f"Folder does not exist:\n{folder}")
            return
        self._open_folder(folder)

    def _run_averaging_pattern(self) -> None:
        if self._running:
            messagebox.showinfo("Averaging Pattern", "Wait until current profiling run is finished.")
            return
        if self._averaging_running:
            messagebox.showinfo("Averaging Pattern", "Averaging is already running.")
            return

        selected_ids = [
            sid for sid, enabled in self.result_selected.items()
            if enabled and sid in self.result_sample_paths
        ]
        if not selected_ids:
            messagebox.showinfo("Averaging Pattern", "No result samples selected. Double-click result rows to mark [x].")
            return

        invalid = [sid for sid in selected_ids if not self.result_validation_ok.get(sid, False)]
        if invalid:
            messagebox.showerror("Averaging Pattern", "Some selected result samples are not validation-ready.")
            return

        channel = str(self.dataset_channel_var.get()).strip().lower()
        output_hint = f"Results/Averaging_YYYYMMDD_HHMMSS (alignment: {channel})"
        preview_lines: list[str] = []
        for sid in selected_ids[:5]:
            row = self.results_tree.item(sid, "values")
            if len(row) >= 3:
                preview_lines.append(f"- {row[1]}/{row[2]}")
        more_count = max(0, len(selected_ids) - 5)
        preview_text = "\n".join(preview_lines)
        if more_count > 0:
            preview_text += f"\n- ... and {more_count} more"

        confirm_msg = (
            f"Selected samples: {len(selected_ids)}\n"
            f"Alignment channel: {channel}\n"
            f"Validation-ready: {len(selected_ids)} / {len(selected_ids)}\n"
            f"Output folder: {output_hint}\n\n"
            f"First selected samples:\n{preview_text}\n\n"
            f"Continue Averaging Pattern?"
        )
        if not messagebox.askyesno("Confirm Averaging Pattern", confirm_msg):
            return

        script_path = self.project_root / "Averaging_Pattern.py"
        if not script_path.exists():
            messagebox.showerror("Averaging Pattern", f"Missing script:\n{script_path}")
            return

        runtime_cfg = self._build_runtime_cfg()
        temp_cfg_path = self.project_root / ".gui_runtime_config_averaging.json"
        with temp_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(runtime_cfg, f, indent=2)

        cmd = [
            sys.executable,
            str(script_path),
            "--config",
            str(temp_cfg_path),
            "--channel",
            channel,
        ]
        for sid in selected_ids:
            cmd.extend(["--sample-dir", str(self.result_sample_paths[sid])])
        self._averaging_running = True
        self.avg_btn.configure(state="disabled")
        self.create_dataset_btn.configure(state="disabled")
        self._log(f"Starting Averaging Pattern for {len(selected_ids)} sample(s)...")
        thread = threading.Thread(
            target=self._run_averaging_worker,
            args=(cmd, temp_cfg_path),
            daemon=True,
        )
        thread.start()

    def _run_averaging_worker(self, cmd: list[str], temp_cfg_path: Path) -> None:
        code = -1
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._avg_proc = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                self._msg_queue.put(line.rstrip())
            code = proc.wait()
        except Exception as exc:
            self._msg_queue.put(f"Averaging Pattern failed: {exc}")
        finally:
            self._avg_proc = None
            try:
                temp_cfg_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._msg_queue.put(f"__AVERAGING_FINISHED__:{code}")

    def _resolve_generated_mask_path(self, sample_dir: Path, sample_name: str, channel_key: str) -> Path | None:
        ch_cap = channel_key.capitalize()
        candidates = [
            sample_dir / "Refinement" / f"{ch_cap}_refined_mask.tiff",
            sample_dir / "Thresholding" / f"{sample_name}_{ch_cap}.tiff",
            sample_dir / "DL_masks" / f"{sample_name}_{ch_cap}.tiff",
        ]
        if channel_key == "fibroblast":
            candidates.extend(
                [
                    sample_dir / "Thresholding" / f"{sample_name}_Fibro.tiff",
                    sample_dir / "DL_masks" / f"{sample_name}_Fibro.tiff",
                ]
            )
        for p in candidates:
            if p.exists():
                return p
        return None

    def _create_dataset_from_results(self) -> None:
        if self._running:
            messagebox.showinfo("Create Dataset", "Wait until current profiling run is finished.")
            return

        channel = str(self.dataset_channel_var.get()).strip().lower()
        if channel not in ("nucleus", "tumor", "fibroblast"):
            messagebox.showerror("Create Dataset", "Please select a valid channel.")
            return

        selected_ids = [
            sid for sid, enabled in self.result_selected.items()
            if enabled and sid in self.result_sample_paths
        ]
        if not selected_ids:
            messagebox.showinfo("Create Dataset", "No result samples selected in Results panel.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = self.project_root / "Training" / f"{channel}_{timestamp}"
        out_root.mkdir(parents=True, exist_ok=True)
        masks_root = out_root / "masks"
        masks_root.mkdir(parents=True, exist_ok=True)
        list_path = out_root / "samples.txt"

        exported = 0
        lines: list[str] = []
        for sid in selected_ids:
            row = self.results_tree.item(sid, "values")
            if len(row) < 5:
                continue
            exp_name = str(row[1])
            sample_name = str(row[2])
            sample_dir = self.result_sample_paths[sid]

            input_dir = sample_dir / "Input"
            input_candidates = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
            if not input_candidates:
                self._log(f"[Create Dataset] Skip {exp_name}/{sample_name}: missing Input TIFF under {input_dir}.")
                continue
            input_tiff = sorted(input_candidates, key=lambda p: p.name.lower())[0]

            stack6 = np.asarray(tifffile.imread(str(input_tiff)))
            if stack6.ndim != 4 or stack6.shape[0] != 6:
                self._log(f"[Create Dataset] Skip {exp_name}/{sample_name}: invalid TIFF shape {stack6.shape}.")
                continue

            raw = {"tumor": stack6[0], "fibroblast": stack6[1], "nucleus": stack6[2]}[channel]

            mask_path = self._resolve_generated_mask_path(sample_dir, sample_name, channel)
            if mask_path is not None:
                mask = _read_tiff_stack_robust(mask_path)
                mask_bin = (np.asarray(mask) > 0).astype(np.uint8) * 255
            else:
                provided = {"tumor": stack6[3], "fibroblast": stack6[4], "nucleus": stack6[5]}[channel]
                mask_bin = (np.asarray(provided) > 0).astype(np.uint8) * 255

            stem = f"{exp_name}_{sample_name}_{channel}"
            raw_out = out_root / f"{stem}_raw.tiff"
            tifffile.imwrite(str(raw_out), np.asarray(raw))

            mask_subdir = masks_root / raw_out.stem
            mask_subdir.mkdir(parents=True, exist_ok=True)
            mask_out = mask_subdir / f"{raw_out.stem}_mask.tiff"
            tifffile.imwrite(str(mask_out), mask_bin)

            lines.append(f"{exp_name}/{sample_name}\n")
            exported += 1

        with list_path.open("w", encoding="utf-8") as f:
            f.writelines(lines)

        self._log(f"Create Dataset: exported {exported} sample(s) to {out_root}")
        if exported == 0:
            messagebox.showwarning("Create Dataset", "No samples exported. Check selected rows and mask availability.")
        else:
            messagebox.showinfo("Create Dataset", f"Exported {exported} sample(s).\n{out_root}")

    def _open_folder(self, folder: Path) -> None:
        if sys.platform.startswith("win"):
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def _run_selected_samples(self) -> None:
        if self._running:
            return
        chosen = [sid for sid, enabled in self.sample_selected.items() if enabled and sid in self.sample_map]
        if not chosen:
            messagebox.showinfo("Run Profiling", "No active samples selected. Double-click sample rows to mark [x].")
            return

        if not messagebox.askyesno("Start Profiling", f"Run profiling for {len(chosen)} selected sample(s)?"):
            return

        runtime_cfg = self._build_runtime_cfg()
        runtime_samples = [self.sample_map[sid] for sid in chosen]

        temp_cfg_path = self.project_root / ".gui_runtime_config.json"
        with temp_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(runtime_cfg, f, indent=2)

        self._running = True
        self._stop_requested = False
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._log(f"Starting profiling for {len(runtime_samples)} selected sample(s).")

        thread = threading.Thread(
            target=self._run_worker,
            args=(temp_cfg_path, runtime_samples),
            daemon=True,
        )
        thread.start()

    def _stop_processing(self) -> None:
        if not self._running:
            return
        self._stop_requested = True
        self._msg_queue.put("Stop requested. Terminating active process...")
        proc = self._active_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _run_worker(self, runtime_cfg_path: Path, runtime_samples: list[SampleDataPaths]) -> None:
        try:
            total = len(runtime_samples)
            for idx, sample in enumerate(runtime_samples, start=1):
                if self._stop_requested:
                    self._msg_queue.put("Processing stopped by user.")
                    break
                sample_id = f"{sample.experiment_name}/{sample.name}"
                self._msg_queue.put(f"[{idx}/{total}] Processing {sample_id}")
                cmd = [
                    sys.executable,
                    str(self.project_root / "Spheroid_Profiling.py"),
                    "--config",
                    str(runtime_cfg_path),
                    "--input-tiff",
                    str(sample.tiff_path),
                    "--experiment",
                    sample.experiment_name,
                    "--sample",
                    sample.name,
                ]
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                self._active_proc = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._msg_queue.put(line.rstrip())
                    if self._stop_requested and proc.poll() is None:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                code = proc.wait()
                self._active_proc = None
                if self._stop_requested:
                    self._msg_queue.put(f"[{idx}/{total}] Stopped {sample_id}")
                    break
                if code != 0:
                    self._msg_queue.put(f"Run failed for {sample_id} (exit code {code})")
                    break
                self._msg_queue.put(f"[{idx}/{total}] Completed {sample_id}")
            if not self._stop_requested:
                self._msg_queue.put("All selected samples completed.")
        except Exception as exc:
            self._msg_queue.put(f"Run failed: {exc}")
        finally:
            self._active_proc = None
            try:
                runtime_cfg_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._msg_queue.put("__RUN_FINISHED__")

    def _drain_messages(self) -> None:
        while not self._msg_queue.empty():
            msg = self._msg_queue.get_nowait()
            if msg == "__RUN_FINISHED__":
                self._running = False
                self.run_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                self.refresh_results_tree()
                continue
            if msg.startswith("__AVERAGING_FINISHED__:"):
                self._averaging_running = False
                self.avg_btn.configure(state="normal")
                self.create_dataset_btn.configure(state="normal")
                code_str = msg.split(":", 1)[1] if ":" in msg else "-1"
                try:
                    code = int(code_str)
                except ValueError:
                    code = -1
                if code == 0:
                    self._log("Averaging Pattern finished.")
                    self.refresh_results_tree()
                else:
                    self._log(f"Averaging Pattern failed (exit code {code}).")
                continue
            if self._should_show_log_line(msg):
                self._log(msg)
        self.after(400, self._drain_messages)

    def _auto_refresh(self) -> None:
        if not self._running:
            self.refresh_samples(silent=True)
            self.refresh_results_tree()
        self.after(5000, self._auto_refresh)

    def _log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    ensure_isolated_runtime(project_root)
    _ensure_runtime_imports()
    config_path = project_root / "Config" / "spheroid_config.json"
    app = SpheroidProfilingGUI(project_root=project_root, config_path=config_path)
    app.mainloop()


if __name__ == "__main__":
    main()

