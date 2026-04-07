"""
Local DL segmentation for spheroid profiling.
Uses networks under Networks/ and local preprocessing logic.
No dependency on Python_Ref, Matlab_Ref, or Python_Post_Ref.

Research group:
Björn Önfelt Group
Department of Applied Physics, Division of Biophysics
Royal Institute of Technology

Coding author:
Hanqing Zhang, Researcher, Royal Institute of Technology, hanzha@kth.se
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is in path for Networks import
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Networks.UNet3D_model import UNet3DSegmentation, get_best_model_path as get_best_model_path_3d
from Networks.DeeplabV3_model import DeepLabV3PlusSegmentation, get_best_model_path as get_best_model_path_2d


def _safe_torch_load(path: Path, device: torch.device):
    """Prefer weights_only loading when supported by current torch."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _normalize(vol: np.ndarray) -> np.ndarray:
    """Normalize intensity volume to [0, 1]."""
    vol = np.asarray(vol, dtype=np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol)
    return vol


def run_dl_segmentation_3d(
    intensity_stack: np.ndarray,
    networks_root: Path,
    class_label_id_foreground: int,
    foreground_value: int,
    roi_size: tuple[int, int, int] = (64, 128, 128),
    overlap: float = 0.25,
    channel_name: str | None = None,
) -> np.ndarray:
    """
    Standalone 3D UNet segmentation using local Networks and weights under Networks/Cells3D.
    Uses sliding-window inference for full-volume prediction.
    Returns binary mask (0 or foreground_value).
    """
    from monai.inferers import sliding_window_inference

    vol = np.asarray(intensity_stack, dtype=np.float32)
    if vol.ndim == 2:
        vol = vol[None, ...]
    vol = _normalize(vol)
    # MONAI sliding_window expects (B, C, D, H, W)
    if vol.ndim == 3:
        vol = vol[None, None, ...]  # (D,H,W) -> (1,1,D,H,W)
    vol_t = torch.from_numpy(vol).float()  # (1, 1, D, H, W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch = channel_name or "channel"
    dev = "GPU" if device.type == "cuda" else "CPU"
    print(f"    Quantifying {ch}: 3D UNet ({dev})")
    params = {
        "Network_InputSize": [roi_size[1], roi_size[2], roi_size[0], 1],  # H, W, D, C
        "Network_classNames": ["bg", "cell"],
    }
    model = UNet3DSegmentation(params).to(device)
    weight_dir = Path(networks_root) / "Cells3D"
    best_path = get_best_model_path_3d(str(weight_dir))
    state = _safe_torch_load(best_path, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    vol_t = vol_t.to(device)
    with torch.no_grad():
        pred = sliding_window_inference(
            inputs=vol_t,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
        )
        pred = torch.argmax(pred, dim=1).squeeze(0)  # (D, H, W)

    pred_np = pred.cpu().numpy().astype(np.int32)
    mask = pred_np == int(class_label_id_foreground)
    return (mask.astype(np.uint8)) * int(foreground_value)


def run_dl_segmentation_2d(
    intensity_stack: np.ndarray,
    networks_root: Path,
    class_label_id_foreground: int,
    foreground_value: int,
    input_hw: tuple[int, int] = (224, 224),
    channel_name: str | None = None,
) -> np.ndarray:
    """
    Standalone 2D DeepLabV3+ segmentation (slice-by-slice) using local Networks
    and weights under Networks/Cells2D.
    Returns binary mask stack (0 or foreground_value).
    """
    vol = np.asarray(intensity_stack, dtype=np.float32)
    if vol.ndim == 2:
        vol = vol[None, ...]
    z, h, w = vol.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch = channel_name or "channel"
    dev = "GPU" if device.type == "cuda" else "CPU"
    print(f"    Quantifying {ch}: 2D DeepLabV3+ ({dev})")
    model = DeepLabV3PlusSegmentation(
        backbone_type="resnet50",
        num_classes=2,
        output_stride=16,
        in_channels=1,
    ).to(device)
    weight_dir = Path(networks_root) / "Cells2D"
    best_path = get_best_model_path_2d(str(weight_dir))
    state = _safe_torch_load(best_path, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    out_mask = np.zeros_like(vol, dtype=np.uint8)
    for zi in range(z):
        sl = _normalize(vol[zi])
        sl_t = torch.from_numpy(sl)[None, None].float()  # (1, 1, H, W)
        sl_t = F.interpolate(sl_t, size=input_hw, mode="bilinear", align_corners=False).to(device)
        with torch.no_grad():
            logits = model(sl_t)
            if isinstance(logits, dict):
                logits = logits["out"]
            pred = torch.argmax(logits, dim=1)[0]  # (Ht, Wt)
        pred_np = pred.cpu().numpy().astype(np.int32)
        pred_t = torch.from_numpy(pred_np)[None, None].float()
        pred_t = F.interpolate(pred_t, size=(h, w), mode="nearest")
        pred_back = pred_t.squeeze(0).squeeze(0).numpy().astype(np.int32)
        out_mask[zi] = (pred_back == int(class_label_id_foreground)).astype(np.uint8) * int(foreground_value)

    return out_mask
