# crowd_detection/main.py
# Detects faces in image frames using YOLOv8.

import json
import time

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

from .config import (
    DEFAULT_CONF, DEFAULT_IOU, MODEL_NAME, PEOPLE_ANNOTATED_DIR, PEOPLE_CLASS_ID, PEOPLE_MODEL_NAME,
    ANNOTATED_DIR, TILE_BATCH, TILE_COLS, TILE_IMGSZ, TILE_OVERLAP, TILE_ROWS, USE_FACE_DETECTION, USE_TILING, SAVE_TILE_DEBUG, TILE_DEBUG_DIR,
    PREDICT_KWARGS, RESOLVED_DEVICE, USE_CUDA,
)
from video_processing.tiling import generate_tiles   # sibling package; see note below

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACE_OUTPUT_DIR = ANNOTATED_DIR if ANNOTATED_DIR.is_absolute() else PROJECT_ROOT / ANNOTATED_DIR
PEOPLE_OUTPUT_DIR = PEOPLE_ANNOTATED_DIR if PEOPLE_ANNOTATED_DIR.is_absolute() else PROJECT_ROOT / PEOPLE_ANNOTATED_DIR
TILE_DEBUG_OUTPUT_DIR = TILE_DEBUG_DIR if TILE_DEBUG_DIR.is_absolute() else PROJECT_ROOT / TILE_DEBUG_DIR
SUMMARY_OUTPUT_DIR = PEOPLE_OUTPUT_DIR.parent   # crowd_detection_output/


def _model_backend(model_path):
    """Rough inference-backend name from the weight path Ultralytics was given."""
    p = str(model_path).lower().rstrip("/\\")
    if p.endswith("_openvino_model") or "openvino" in p:
        return "openvino"
    if p.endswith(".onnx"):
        return "onnx"
    if p.endswith((".engine", ".plan")):
        return "tensorrt"
    return "pytorch"


def _next_run_number(out_dir):
    """Next detection_summary_run_NNN.json index in out_dir (1 if none exist).
    Footage is re-run repeatedly, so summaries are numbered per run, not per
    video; the video id is still recorded inside each file."""
    highest = 0
    for path in out_dir.glob("detection_summary_run_*.json"):
        tail = path.stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            highest = max(highest, int(tail))
    return highest + 1


def _safe_video_id(video_id):
    value = str(video_id or "unknown_video")
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)

def _iou(box, boxes):
    ix1 = np.maximum(box[0], boxes[:, 0]); iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2]); iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area  = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)

def _nms(boxes, scores, iou_thresh):
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]; keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[_iou(boxes[i], boxes[rest]) < iou_thresh]
    return keep


def load_models():
    face_model = None
    if USE_FACE_DETECTION:
        print(f"[INFO] Loading face model: {MODEL_NAME}")
        face_model = YOLO(MODEL_NAME)

    print(f"[INFO] Loading people model: {PEOPLE_MODEL_NAME}")
    # task="detect" so an OpenVINO/ONNX export dir loads without the task-guess warning
    people_model = YOLO(PEOPLE_MODEL_NAME, task="detect")

    # Move .pt weights onto the GPU up front so the first frame isn't paying the
    # host->device copy. A TensorRT .engine is already device-bound; skip it.
    if USE_CUDA and str(PEOPLE_MODEL_NAME).lower().endswith(".pt"):
        people_model.to(RESOLVED_DEVICE)
    print(f"[INFO] Inference device: {RESOLVED_DEVICE}")

    print("[INFO] Models ready ✓\n")
    return face_model, people_model

def _save_tile_debug(tile, m, tile_dets, out_dir, tag, rows, cols, border_pad):
    """Write one tile crop with its own detections drawn, in tile-local
    coordinates: green = kept, red = dropped as a seam duplicate. Red lines mark
    the border_pad band on each inner seam (the edges the drop rule applies to).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img = tile.copy()                         # tiles are views into the frame
    th_, tw_ = img.shape[:2]

    if m["column"] > 0:
        cv2.line(img, (border_pad, 0), (border_pad, th_), (0, 0, 255), 1)
    if m["column"] < cols - 1:
        cv2.line(img, (tw_ - border_pad, 0), (tw_ - border_pad, th_), (0, 0, 255), 1)
    if m["row"] > 0:
        cv2.line(img, (0, border_pad), (tw_, border_pad), (0, 0, 255), 1)
    if m["row"] < rows - 1:
        cv2.line(img, (0, th_ - border_pad), (tw_, th_ - border_pad), (0, 0, 255), 1)

    kept = 0
    for x1, y1, x2, y2, score, is_kept in tile_dets:
        colour = (0, 200, 0) if is_kept else (0, 0, 255)
        kept += is_kept
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
        cv2.putText(img, f"{score:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    cv2.putText(img, f"r{m['row']}c{m['column']}  kept {kept}/{len(tile_dets)}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out_dir / f"{tag}_tile_r{m['row']}c{m['column']}.jpg"), img)


def detect_tiled(model, frame, conf, iou, keep_class=None,
                 rows=TILE_ROWS, cols=TILE_COLS, overlap=TILE_OVERLAP, border_pad=4, imgsz=TILE_IMGSZ,
                 batch=TILE_BATCH, debug_dir=None, debug_tag="frame"):
    """Tile-and-merge detection. Returns the same [{'bbox', 'confidence'}] list
    the whole-frame path returns, so draw_boxes() is unchanged.

    Tiles are run in batched forward passes (batch tiles per call) rather than
    one call per tile; this is where GPU acceleration actually pays off, and it
    helps on CPU too by amortising per-call overhead.

    If debug_dir is given, each tile crop is written there with its own
    detections drawn (green kept, red dropped) for inspecting the tiling.
    """
    h, w = frame.shape[:2]
    tiles, meta = generate_tiles(frame, rows, cols, overlap)

    boxes, scores = [], []
    for start in range(0, len(tiles), batch):
        chunk_tiles = tiles[start:start + batch]
        chunk_meta = meta[start:start + batch]
        results = model(chunk_tiles, conf=conf, iou=iou, imgsz=imgsz, verbose=False, **PREDICT_KWARGS)

        for tile, m, r in zip(chunk_tiles, chunk_meta, results):
            tw, th = m["width"], m["height"]
            tile_dets = []   # (x1, y1, x2, y2, score, kept) in tile-local coords, for debug
            for b in r.boxes:
                if keep_class is not None and int(b.cls[0]) != keep_class:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                score = float(b.conf[0])

                # drop a box glued to an INNER seam; the neighbouring tile,
                # whose overlap strip holds the whole object, keeps the good one
                on_seam = ((x1 <= border_pad and m["column"] > 0) or
                           (x2 >= tw - border_pad and m["column"] < cols - 1) or
                           (y1 <= border_pad and m["row"] > 0) or
                           (y2 >= th - border_pad and m["row"] < rows - 1))
                tile_dets.append((x1, y1, x2, y2, score, not on_seam))
                if on_seam:
                    continue

                # remap tile-local -> full frame (plain offset; tiles aren't resized)
                boxes.append((x1 + m["x"], y1 + m["y"], x2 + m["x"], y2 + m["y"]))
                scores.append(score)

            if debug_dir is not None:
                _save_tile_debug(tile, m, tile_dets, Path(debug_dir), debug_tag,
                                 rows, cols, border_pad)

    if not boxes:
        return []

    boxes  = np.clip(np.array(boxes, float), 0, [w, h, w, h])   # clamp to frame
    scores = np.array(scores, float)
    return [
        {"bbox": [int(v) for v in boxes[i]],
         "confidence": round(float(scores[i]), 4)}
        for i in _nms(boxes, scores, iou)                       # cross-tile dedupe
    ]

def detect_faces(model, frame, conf, iou, use_tiling=False, debug_dir=None, debug_tag="frame"):
    if use_tiling:
        return detect_tiled(model, frame, conf, iou,             # single class
                            debug_dir=debug_dir, debug_tag=f"{debug_tag}_face")
    results = model(frame, conf=conf, iou=iou, verbose=False, **PREDICT_KWARGS)[0]
    return [{"bbox": list(map(int, b.xyxy[0].tolist())),
             "confidence": round(float(b.conf[0]), 4)} for b in results.boxes]

def detect_people(model, frame, conf, iou, use_tiling=False, debug_dir=None, debug_tag="frame"):
    if use_tiling:
        return detect_tiled(model, frame, conf, iou, keep_class=PEOPLE_CLASS_ID ,   # COCO person
                            debug_dir=debug_dir, debug_tag=f"{debug_tag}_people")
    results = model(frame, conf=conf, iou=iou, verbose=False, **PREDICT_KWARGS)[0]
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])

        # COCO class 0 = person
        if cls != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": round(float(box.conf[0]), 4),
        })

    return detections

def draw_people_boxes(frame, detections):
    output = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]

        # Blue boxes for people
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 100, 0), 2)

        label = f"{d['confidence']:.2f}"
        cv2.putText(output, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 100, 0), 1)
  
    cv2.putText(output, f"People: {len(detections)}", (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

    return output

def draw_boxes(frame, detections):
    output = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]

        # Draw bounding box around face
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 80), 2)

        # Draw confidence score above the box
        label = f"{d['confidence']:.2f}"
        cv2.putText(output, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 80), 1)

    # Draw total face count in top left corner
    cv2.putText(output, f"Faces: {len(detections)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 100), 2)

    return output


def detect_crowd(processed_video: dict) -> dict:
    face_model, people_model = load_models() 
    all_results = []
    safe_video_id = _safe_video_id(processed_video.get("video_id"))
    people_video_output_dir = PEOPLE_OUTPUT_DIR / safe_video_id
    frame_width = int(processed_video.get("frame_width") or 0)
    frame_height = int(processed_video.get("frame_height") or 0)
    
    # create output folder
    if USE_FACE_DETECTION:
        FACE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    people_video_output_dir.mkdir(parents=True, exist_ok=True)

    tile_debug_dir = None
    if USE_TILING and SAVE_TILE_DEBUG:
        tile_debug_dir = TILE_DEBUG_OUTPUT_DIR / safe_video_id

    detection_ms_total = 0.0

    for frame_data in processed_video["frames"]:
        frame_path = frame_data["frame_path"]
        resolved_frame_path = Path(frame_path)
        if not resolved_frame_path.is_absolute():
            resolved_frame_path = PROJECT_ROOT / resolved_frame_path

        frame = cv2.imread(str(resolved_frame_path))

        if frame is None:
            print(f"[WARN] Could not read frame {frame_data['frame_id']} — skipping")
            continue

        if frame_width <= 0 or frame_height <= 0:
            frame_height, frame_width = frame.shape[:2]

        debug_tag = f"frame_{frame_data['frame_id']:04d}"

        # time the detection work only (model inference + tiling/NMS), not the
        # frame read or the annotate/save I/O
        detect_start = time.perf_counter()
        people_detections = detect_people(people_model, frame, DEFAULT_CONF, DEFAULT_IOU,
                                          use_tiling=USE_TILING, debug_dir=tile_debug_dir, debug_tag=debug_tag)

        # face detection is an optional, separate output stream; nothing
        # downstream consumes it, so it is off by default (USE_FACE_DETECTION)
        face_detections = None
        face_count = None
        face_annotated_frame_path = None
        if USE_FACE_DETECTION:
            face_detections = detect_faces(face_model, frame, DEFAULT_CONF, DEFAULT_IOU,
                                           use_tiling=USE_TILING, debug_dir=tile_debug_dir, debug_tag=debug_tag)
        detection_ms = round((time.perf_counter() - detect_start) * 1000, 1)
        detection_ms_total += detection_ms

        if USE_FACE_DETECTION:
            face_count = len(face_detections)
            annotated = draw_boxes(frame, face_detections)
            face_output_path = FACE_OUTPUT_DIR / f"frame_{frame_data['frame_id']:04d}.jpg"
            cv2.imwrite(str(face_output_path), annotated)
            face_annotated_frame_path = str(face_output_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

        # save annotated frame for people
        people_annotated = draw_people_boxes(frame, people_detections)
        people_output_path = people_video_output_dir / f"frame_{frame_data['frame_id']:04d}.jpg"
        cv2.imwrite(str(people_output_path), people_annotated)
        people_annotated_frame_path = str(people_output_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

        all_results.append({
            "frame_id": frame_data["frame_id"],
            "timestamp": frame_data["timestamp"],
            "frame_path": frame_path,
            "face_annotated_frame_path": face_annotated_frame_path,
            "people_annotated_frame_path": people_annotated_frame_path,
            "person_count": len(people_detections),
            "face_count": face_count,
            "detection_ms": detection_ms,
            "face_detections": face_detections if face_detections is not None else [],
            "people_detections": people_detections,
        })

    frames_timed = len(all_results)
    peak_people = max((f["person_count"] for f in all_results), default=0)
    backend = _model_backend(PEOPLE_MODEL_NAME)

    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_number = _next_run_number(SUMMARY_OUTPUT_DIR)

    summary = {
        "run": run_number,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_id": processed_video["video_id"],
        "model": Path(str(PEOPLE_MODEL_NAME).rstrip("/\\")).name,
        "backend": backend,
        "device": RESOLVED_DEVICE,
        "openvino": backend == "openvino",
        "tiling": USE_TILING,
        "frames_processed": frames_timed,
        "total_detection_seconds": round(detection_ms_total / 1000, 2),
        "peak_people_per_frame": peak_people,
    }

    summary_path = SUMMARY_OUTPUT_DIR / f"detection_summary_run_{run_number:03d}.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== crowd detection summary (run {run_number}) ===")
    print(f"  model           : {summary['model']}")
    print(f"  backend/device  : {summary['backend']} on {summary['device']}")
    print(f"  openvino        : {summary['openvino']}")
    print(f"  total time      : {summary['total_detection_seconds']:.2f} s over {frames_timed} frame(s)")
    print(f"  people detected : peak {peak_people}/frame")
    print(f"  summary json    : {summary_path}")

    return {
        "video_id": processed_video["video_id"],
        "frame_width": frame_width,
        "frame_height": frame_height,
        "detection_summary": summary,
        "frames":   all_results,
    }

    
    
