"""Shared integration service for fire and stampede detection."""

from fire_detection.main import process_video as process_fire
from stampede_detection.main import analyze_stampede


def process_safety_detection(
    video_data: dict,
    frame_data: dict,
    camera_id: str = "CAM_02",
    zone_id: str = "ZONE_B",
) -> dict:
    """
    Run fire and stampede detection for the same video.

    Fire detection uses the already-extracted frame metadata.
    Stampede detection uses the original video path and performs
    its own consecutive-frame extraction.
    """

    # Fire detection needs extracted frames.
    fire_input = {
        "video_id": video_data.get("video_id"),
        "video_path": video_data.get("video_path"),
        "frames": frame_data.get("frames", []),
    }

    fire_result = process_fire(
        frame_metadata=fire_input,
        camera_id=camera_id,
        zone_id=zone_id,
    )

    # Stampede detection intentionally uses the original video.
    stampede_result = analyze_stampede(
        input_source=video_data,
        camera_id=camera_id,
        zone_id=zone_id,
    )

    # Data aggregation expects "detections", while Stampede
    # Detection currently returns its results as "events".
    normalized_stampede = {
        "video_id": stampede_result.get(
            "video_id",
            video_data.get("video_id"),
        ),
        "detections": stampede_result.get("events", []),
    }

    return {
        "video_id": video_data.get("video_id"),
        "fire": fire_result,
        "stampede": normalized_stampede,
    }
