import httpx
from fastapi import HTTPException, UploadFile
from app.config import USE_MOCK_SERVICES, PLAYER_SERVICE_URL


def get_mock_player_data():
    return {
        "players": [
            {
                "player_id": 1,
                "team": "Team A",
                "position": {"x": 120, "y": 340},
                "speed": 6.4,
                "distance_covered": 3.2,
                "sprints": 4
            },
            {
                "player_id": 2,
                "team": "Team B",
                "position": {"x": 210, "y": 280},
                "speed": 5.8,
                "distance_covered": 2.9,
                "sprints": 3
            }
        ],
        "heatmap": None
    }


async def get_player_data(file: UploadFile = None):
    if USE_MOCK_SERVICES:
        return get_mock_player_data()

    if file is None:
        raise HTTPException(status_code=400, detail="Missing video file")

    try:
        contents = await file.read()

        files = {
            "file": (file.filename, contents, file.content_type)
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{PLAYER_SERVICE_URL}/tracking",
                files=files
            )

        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Player tracking service error: {exc.response.text}"
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=500,
            detail="Could not connect to player tracking service"
        )