import os
from pathlib import Path

CURRENT_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.path.join(CURRENT_DIR, "face_model.pt")   # Model downloaded from https://huggingface.co/arnabdhar/YOLOv8-Face-Detection

# --- Inference device -------------------------------------------------------
# Set CROWD_DEVICE to pick the backend without editing this file:
#   "cpu"    -> OpenVINO IR dir below (Intel-CPU optimised, ~2-4x over plain CPU torch)
#   "cuda:0" -> NVIDIA GPU; uses the TensorRT .engine if one has been built,
#               otherwise the plain .pt weights on CUDA
#   "auto"   -> cuda:0 when a CUDA build of torch sees a GPU, else cpu  (default)
# Needs a CUDA build of torch for the GPU paths:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
DEVICE = os.environ.get("CROWD_DEVICE", "auto")


def _resolve_device(pref):
    pref = (pref or "auto").strip().lower()
    if pref in ("", "auto"):
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if pref.startswith("cuda") or pref.isdigit():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CROWD_DEVICE={pref!r} requested but torch.cuda.is_available() is False. "
                "Install a CUDA build of torch, e.g.\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
            )
        return f"cuda:{pref}" if pref.isdigit() else pref
    return pref  # "cpu", "mps", ...


RESOLVED_DEVICE = _resolve_device(DEVICE)
USE_CUDA = RESOLVED_DEVICE.startswith("cuda")

# OpenVINO export of yolo26mcrowdpeoplefaces.pt (dynamic batch, imgsz 640) - ~2-4x
# faster on CPU, same weights. Re-export with:
#   YOLO("yolo26mcrowdpeoplefaces.pt").export(format="openvino", imgsz=640, dynamic=True)
# TensorRT engine for NVIDIA (machine/GPU-specific, rebuild per box, do not commit):
#   yolo export model=yolo26mcrowdpeoplefaces.pt format=engine imgsz=640 half=True dynamic=True device=0
_PEOPLE_PT       = os.path.join(CURRENT_DIR, "yolo26mcrowdpeoplefaces.pt")
_PEOPLE_ENGINE   = os.path.join(CURRENT_DIR, "yolo26mcrowdpeoplefaces.engine")
_PEOPLE_OPENVINO = os.path.join(CURRENT_DIR, "yolo26mcrowdpeoplefaces_openvino_model")

if USE_CUDA:
    PEOPLE_MODEL_NAME = _PEOPLE_ENGINE if os.path.isfile(_PEOPLE_ENGINE) else _PEOPLE_PT
else:
    PEOPLE_MODEL_NAME = _PEOPLE_OPENVINO

# Spread into every model(...) predict call so device selection is explicit
# (never a silent CPU fallback) and FP16 is used on GPU.
PREDICT_KWARGS = {"device": RESOLVED_DEVICE}
if USE_CUDA:
    PREDICT_KWARGS["half"] = True   # ~2x throughput, negligible effect on detection

PEOPLE_CLASS_ID   = 1

ANNOTATED_DIR = Path("crowd_detection_output") / "face_detection_results"
PERSON_CLASS = None
PEOPLE_ANNOTATED_DIR = Path("crowd_detection_output") / "people_detection_results"

DEFAULT_CONF = 0.20
DEFAULT_IOU  = 0.30
USE_TILING = True
USE_FACE_DETECTION = False
SAVE_TILE_DEBUG = False
TILE_DEBUG_DIR = Path("crowd_detection_output") / "tile_debug"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


OUTPUT_DIR = Path("detection_output")

TILE_ROWS = 4
TILE_COLS = 6
TILE_OVERLAP = 0.2
TILE_IMGSZ = 640

# Tiles run in batched forward passes; a GPU chews through a bigger batch.
TILE_BATCH = 16 if USE_CUDA else 8
