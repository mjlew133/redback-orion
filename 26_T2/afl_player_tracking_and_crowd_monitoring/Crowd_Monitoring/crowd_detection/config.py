import os
from pathlib import Path

CURRENT_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.path.join(CURRENT_DIR, "face_model.pt")   # Model downloaded from https://huggingface.co/arnabdhar/YOLOv8-Face-Detection

# --- Inference device -------------------------------------------------------
# Set CROWD_DEVICE to pick the backend without editing this file:
#   "cpu"    -> OpenVINO IR dir below (Intel-CPU optimised, ~2-4x over plain CPU torch)
#   "cuda:0" -> NVIDIA GPU; uses the TensorRT .engine if one has been built,
#               otherwise the plain .pt weights on CUDA
#   "dml"    -> DirectML: the .onnx export on any DX12 GPU (AMD / Intel). Needs
#               `pip install onnxruntime-directml` and NOTHING else named
#               onnxruntime* (plain onnxruntime / -gpu share a folder and shadow
#               it). Build the export with:
#                 yolo export model=yolo26mcrowdpeoplefaces.pt format=onnx imgsz=640 dynamic=True opset=17
#   "auto"   -> cuda:0 when a CUDA build of torch sees a GPU, else cpu  (default)
# Needs a CUDA build of torch for the CUDA paths:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# (use cu128, not cu124 — cu124 has no wheels for Python 3.14; cu128 does.
# If nvidia-smi shows an older driver that cu128 rejects at runtime, try cu126.)
DEVICE = os.environ.get("CROWD_DEVICE", "dml")

## CHANGE AUTO DML
def _resolve_device(pref):
    pref = (pref or "auto").strip().lower()
    if pref in ("", "auto"):
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if pref == "dml":
        # Ultralytics' check_requirements would otherwise pip-install plain
        # onnxruntime over the DirectML build. Must be set before `import
        # ultralytics`, so config has to be imported before it (see main.py).
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        return "dml"
    if pref.startswith("cuda") or pref.isdigit():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CROWD_DEVICE={pref!r} requested but torch.cuda.is_available() is False. "
                "Install a CUDA build of torch, e.g.\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
            )
        return f"cuda:{pref}" if pref.isdigit() else pref
    return pref  # "cpu", "mps", ...


RESOLVED_DEVICE = _resolve_device(DEVICE)
USE_CUDA = RESOLVED_DEVICE.startswith("cuda")
USE_DML  = RESOLVED_DEVICE == "dml"

# OpenVINO export of yolo26mcrowdpeoplefaces.pt (dynamic batch, imgsz 640) - ~2-4x
# faster on CPU, same weights. Re-export with:
#   YOLO("yolo26mcrowdpeoplefaces.pt").export(format="openvino", imgsz=640, dynamic=True)
# TensorRT engine for NVIDIA (machine/GPU-specific, rebuild per box, do not commit):
#   yolo export model=yolo26mcrowdpeoplefaces.pt format=engine imgsz=640 quantize=16 dynamic=True device=0
_PEOPLE_PT       = os.path.join(CURRENT_DIR, "yolo26mcrowdpeoplefaces.pt")
_PEOPLE_ENGINE   = os.path.join(CURRENT_DIR, "yolo26mcrowdpeoplefaces.engine")
_PEOPLE_ONNX     = os.path.join(CURRENT_DIR, "yolo26mcrowdpeoplefaces.onnx")
_PEOPLE_OPENVINO = os.path.join(CURRENT_DIR, "yolo26mcrowdpeoplefaces_openvino_model")

if USE_CUDA:
    PEOPLE_MODEL_NAME = _PEOPLE_ENGINE if os.path.isfile(_PEOPLE_ENGINE) else _PEOPLE_PT
elif USE_DML:
    PEOPLE_MODEL_NAME = _PEOPLE_ONNX
else:
    PEOPLE_MODEL_NAME = _PEOPLE_OPENVINO

# Spread into every model(...) predict call so device selection is explicit
# (never a silent CPU fallback) and FP16 is used on GPU.
PREDICT_KWARGS = {"device": RESOLVED_DEVICE}
if USE_CUDA:
    PREDICT_KWARGS["quantize"] = 16   # FP16 (replaces the deprecated half=True): ~2x throughput, negligible effect on detection
elif USE_DML:
    # Ultralytics can't parse "dml" as a torch device. The ONNX session itself
    # runs on DirectML (dml_backend.enable_dml repins it); Ultralytics only needs
    # a valid torch device for its pre/post-processing tensors.
    PREDICT_KWARGS["device"] = "cpu"

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

TILE_ROWS = 2
TILE_COLS = 3
TILE_OVERLAP = 0.2
TILE_IMGSZ = 640

# Tiles run in batched forward passes; a GPU chews through a bigger batch.
TILE_BATCH = 16 if (USE_CUDA or USE_DML) else 8

# --- Latency knobs (crowd detection only) ---------------------------------
# All three are safe to leave at the values below (= current behaviour) and
# can be swept from the environment without editing this file.

# 1. Temporal decimation. Run the detector on every Nth extracted frame and
#    carry the previous result forward for the rest. 1 = detect every frame.
#    Crowd counts barely move frame-to-frame, so 3-8 is usually invisible.
DETECT_STRIDE = int(os.environ.get("CROWD_DETECT_STRIDE", "8"))

# 2. Downscale the frame before tiling/inference (boxes are scaled back to
#    full res afterwards). 0 = off. 1920 turns 4K into ~1080p - ~4x fewer
#    pixels - at some cost to distant-head recall, so pair it with fewer
#    tiles (e.g. TILE_ROWS=2, TILE_COLS=3).
DETECT_MAX_WIDTH = int(os.environ.get("CROWD_DETECT_MAX_WIDTH", "1920"))

# 3. Skip tiles that are almost entirely black - the field pixels
#    crowd_region_preprocessing already masked out. Free: no people there.
SKIP_EMPTY_TILES = os.environ.get("CROWD_SKIP_EMPTY_TILES", "true").strip().lower() in {"1", "true", "yes", "on"}
MIN_TILE_FILL = 0.02   # a tile needs at least this fraction of non-black pixels to be worth running
