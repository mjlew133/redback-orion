# pipelines/pipeline_cluster.py

from team_clusterer import TeamClusterer
from color_utils import identify_referee
import cv2

def run_cluster_pipeline(frames_data, reid_model, processor):

    clusterer = TeamClusterer()

    for frame_idx, frame in enumerate(processor.read_video()):
        detections = frames_data.get(frame_idx, [])

        crops = []
        valid_dets = []

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            crop = frame[y1:y2, x1:x2]

            if crop is None or crop.size == 0:
                continue

            crops.append(crop)
            valid_dets.append(det)

        embeddings = reid_model.extract_embeddings_batch(crops)

        if len(embeddings) == 0:
            continue

        labels = clusterer.cluster(embeddings)
        ref_cluster = identify_referee(labels)

        for det, label in zip(valid_dets, labels):

            if label == -1:
                text = "unknown"
            elif label == ref_cluster:
                text = "referee"
            else:
                text = f"team_{label}"

            x1, y1, x2, y2 = map(int, det['bbox'])

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, text, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        processor.writer.write(frame)