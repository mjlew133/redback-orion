"""Shared integration service for fire and stampede detection."""

from fire_detection.main import process_video as process_fire
from stampede_detection.main import analyze_stampede


def process_safety_detection(
    video_data: dict,
    camera_id: str = "CAM_02",
    zone_id: str = "ZONE_B",
) -> dict:

    # 1. Fire detection
    fire_result = process_fire(
        frame_metadata=video_data,
        camera_id=camera_id,
        zone_id=zone_id,
    )
    # 2. Stampede detection
    stampede_result = analyze_stampede(
        input_source=video_data,
        camera_id=camera_id,
        zone_id=zone_id,
    )
    # 3. Normalize stampede output
    normalized_stampede = {
        "video_id": stampede_result.get(
            "video_id",
            video_data.get("video_id"),
        ),
        "detections": stampede_result.get("detections", []),
    }

    return {
        "video_id": video_data.get("video_id"),
        "fire": fire_result,
        "stampede": normalized_stampede,
    }
