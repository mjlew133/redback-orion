# Throwaway timing harness. Times every stage of the pipeline without
# editing pipeline source: it monkey-patches the functions in place, runs
# all three stages, then prints a breakdown.
#
# Run from crowd_monitoring/:   python -m time_split
#
# Nothing here is imported by the pipeline. Delete it when you are done.

import time
import functools
import threading
from collections import defaultdict

import cv2

import video_processing.main as vp
import crowd_detection.main as cd
import crowd_region_preprocessing.main as crp


VIDEO_ID = "match_02"
VIDEO_PATH = "data/raw/match_02.mp4"


# ----------------------------------------------------------------------
# timer registry
# ----------------------------------------------------------------------

_lock = threading.Lock()
_total = defaultdict(float)     # label -> seconds
_count = defaultdict(int)       # label -> calls
_extra = {}                     # free-form counters


def _add(label, seconds):
    with _lock:
        _total[label] += seconds
        _count[label] += 1


class _timed:
    """Context manager and the primitive everything else is built on."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _add(self.label, time.perf_counter() - self.t0)
        return False


def _wrap(module, name, label=None):
    """Replace module.name with a timed version of itself."""
    original = getattr(module, name)
    label = label or name

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        with _timed(label):
            return original(*args, **kwargs)

    setattr(module, name, wrapper)
    return original


# ----------------------------------------------------------------------
# cv2 patches
#
# All three pipeline modules do "import cv2" and resolve the attribute at
# call time, so patching here catches every call site in all of them.
# ----------------------------------------------------------------------

_wrap(cv2, "imread", "cv2.imread")
_wrap(cv2, "imwrite", "cv2.imwrite")
_wrap(cv2, "resize", "cv2.resize")
_wrap(cv2, "cvtColor", "cv2.cvtColor")
_wrap(cv2, "countNonZero", "cv2.countNonZero")
_wrap(cv2, "bitwise_and", "cv2.bitwise_and (mask apply)")

_RealVideoCapture = cv2.VideoCapture


class _TimedVideoCapture:
    """Proxy so cap.read() can be timed. This is the video decode cost,
    which is otherwise invisible and is paid twice: once by
    get_video_stats and once by the main extraction loop."""

    def __init__(self, *args, **kwargs):
        with _timed("cv2.VideoCapture.open"):
            self._cap = _RealVideoCapture(*args, **kwargs)

    def read(self, *args, **kwargs):
        with _timed("cv2.VideoCapture.read"):
            return self._cap.read(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._cap, name)


cv2.VideoCapture = _TimedVideoCapture


# ----------------------------------------------------------------------
# video_processing patches
# ----------------------------------------------------------------------

_wrap(vp, "get_video_stats", "vp.get_video_stats (blur threshold pass)")
_wrap(vp, "check_blur", "vp.check_blur")
_wrap(vp, "save_frame_worker", "vp.save_frame_worker (THREADED)")
_wrap(vp, "generate_tiles", "vp.generate_tiles")
_wrap(vp, "runtime_preprocessing", "vp.runtime_preprocessing")
_wrap(vp, "load_config", "vp.load_config")


# ----------------------------------------------------------------------
# crowd_region_preprocessing patches
#
# Wrapped defensively: helper names may differ, so missing ones are
# skipped rather than crashing the harness.
# ----------------------------------------------------------------------

for _name, _label in (("_build_keep_mask", "crp._build_keep_mask"),
                      ("_normalize_polygons", "crp._normalize_polygons"),
                      ("load_config", "crp.load_config")):
    if hasattr(crp, _name):
        _wrap(crp, _name, _label)


# ----------------------------------------------------------------------
# crowd_detection patches
# ----------------------------------------------------------------------

_wrap(cd, "tiles_from_meta", "cd.tiles_from_meta")
_wrap(cd, "draw_people_boxes", "cd.draw_people_boxes")
_wrap(cd, "_nms", "cd._nms (cross-tile dedupe)")

_real_detect_tiled = cd.detect_tiled
_real_detect_people = cd.detect_people
_real_load_models = cd.load_models


class _TimedModel:
    """Wraps a YOLO instance so the forward pass is timed, and harvests
    Ultralytics' own preprocess/inference/postprocess split from the
    results. Preprocess is the letterbox resize up to imgsz.

    NOTE: the yolo.* labels below are nested inside the forward pass,
    which is nested inside detect_tiled, which is nested inside
    detect_people. Four labels, one span of time."""

    def __init__(self, model, label):
        self._model = model
        self._label = label

    def __call__(self, *args, **kwargs):
        with _timed(f"{self._label} forward pass"):
            results = self._model(*args, **kwargs)
        try:
            for r in results:
                speed = getattr(r, "speed", None) or {}
                _add("yolo preprocess (letterbox) [nested]", speed.get("preprocess", 0.0) / 1000.0)
                _add("yolo inference [nested]", speed.get("inference", 0.0) / 1000.0)
                _add("yolo postprocess (per-tile nms) [nested]", speed.get("postprocess", 0.0) / 1000.0)
        except TypeError:
            pass
        return results

    def __getattr__(self, name):
        return getattr(self._model, name)


def _timed_load_models():
    with _timed("cd.load_models"):
        face_model, people_model = _real_load_models()
    return (
        _TimedModel(face_model, "face model") if face_model is not None else None,
        _TimedModel(people_model, "people model"),
    )


def _timed_detect_tiled(tiles, meta, *args, **kwargs):
    _extra["tile_grid"] = (max(m["row"] for m in meta) + 1,
                           max(m["column"] for m in meta) + 1)
    _extra["tiles_in"] = _extra.get("tiles_in", 0) + len(tiles)
    _extra["tiles_per_frame"] = len(tiles)
    if tiles:
        _extra["tile_size"] = (meta[0]["width"], meta[0]["height"])
    _extra["tiled_calls"] = _extra.get("tiled_calls", 0) + 1
    with _timed("cd.detect_tiled (total)"):
        out = _real_detect_tiled(tiles, meta, *args, **kwargs)
    _extra["kept_boxes"] = _extra.get("kept_boxes", 0) + len(out)
    return out


def _timed_detect_people(*args, **kwargs):
    if not kwargs.get("use_tiling"):
        _extra["untiled_frames"] = _extra.get("untiled_frames", 0) + 1
    with _timed("cd.detect_people (total)"):
        return _real_detect_people(*args, **kwargs)


cd.load_models = _timed_load_models
cd.detect_tiled = _timed_detect_tiled
cd.detect_people = _timed_detect_people


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------

print("=" * 72)
t0 = time.perf_counter()
processed = vp.process_video(VIDEO_ID, VIDEO_PATH)
t1 = time.perf_counter()

if "error" in processed:
    raise SystemExit(processed["error"])

# Builds the keep-mask and attaches it as processed_video["crowd_mask"].
# Without this stage detect_crowd gets no mask, every tile stays non-black,
# and skip_empty drops nothing.
prepared = crp.prepare_crowd_frames(processed)
t2 = time.perf_counter()

result = cd.detect_crowd(prepared)
t3 = time.perf_counter()

wall = t3 - t0
summary = result["detection_summary"]


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------

def _line(label, seconds, calls=None, of=None):
    pct = f"{seconds / of * 100:5.1f}%" if of else "      "
    per = f"{seconds / calls * 1000:8.2f}" if calls else "        "
    calls_s = f"{calls:>7}" if calls else "       "
    print(f"  {label:<42} {seconds:8.2f}s {pct} {calls_s} {per} ms")


print()
print("=" * 72)
print("TOP LEVEL")
print("=" * 72)
_line("process_video", t1 - t0, of=wall)
_line("prepare_crowd_frames", t2 - t1, of=wall)
_line("detect_crowd", t3 - t2, of=wall)
_line("  of which timed detection", summary["total_detection_seconds"], of=wall)
_line("  of which everything else", (t3 - t2) - summary["total_detection_seconds"], of=wall)
print(f"  {'TOTAL WALL CLOCK':<42} {wall:8.2f}s")

print()
print("=" * 72)
print(f"{'BY FUNCTION':<44} {'total':>9} {'share':>6} {'calls':>7} {'per call':>11}")
print("=" * 72)
for label in sorted(_total, key=_total.get, reverse=True):
    _line(label, _total[label], _count[label], of=wall)
print()
print("  Labels marked [nested] sit inside 'people model forward pass',")
print("  which sits inside detect_tiled, which sits inside detect_people.")
print("  Compare siblings, never sum the column.")

print()
print("=" * 72)
print("CONTEXT")
print("=" * 72)
print(f"  frames extracted        : {len(processed['frames'])}")
print(f"  frames detected         : {summary['frames_detected']} (stride {summary['detect_stride']})")
print(f"  frame size              : {processed['frame_width']}x{processed['frame_height']}")
print(f"  mask source             : {prepared.get('crowd_mask_source', 'NONE - no mask attached')}")

mask = prepared.get("crowd_mask")
if mask is not None:
    import numpy as np
    kept_frac = float(np.count_nonzero(mask)) / mask.size
    print(f"  mask keeps              : {kept_frac * 100:.1f}% of frame area")

if "tile_grid" in _extra:
    rows, cols = _extra["tile_grid"]
    tw, th = _extra.get("tile_size", (0, 0))
    tiled_calls = _extra.get("tiled_calls", 0)
    tiles_in = _extra.get("tiles_in", 0)
    passes = _count.get("yolo inference [nested]", 0)
    print(f"  tile grid               : {rows}x{cols} = {_extra['tiles_per_frame']} tiles/frame")
    print(f"  tile size               : {tw}x{th} px")
    print(f"  frames tiled            : {tiled_calls}")
    print(f"  tiles handed to detector: {tiles_in}")
    print(f"  forward passes actually : {passes}")
    if tiles_in:
        skipped = tiles_in - passes
        print(f"  tiles skipped (empty)   : {skipped} ({skipped / tiles_in * 100:.1f}%)")
        if skipped == 0:
            print("     -> skip_empty dropped nothing. Either no mask, or")
            print("        MIN_TILE_FILL is too low to catch partial tiles.")
else:
    print("  tile grid               : NONE - detection ran on whole frames")
print(f"  frames NOT tiled        : {_extra.get('untiled_frames', 0)}")
print(f"  peak people/frame       : {summary['peak_people_per_frame']}")

print()
print("  NOTE: save_frame_worker runs in a 4-thread pool, so its summed time")
print("  overlaps the main thread and will exceed its share of wall clock.")
print()