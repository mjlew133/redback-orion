import cv2
import os
import json
import numpy as np                     # Used for creating the "canvas" for letterboxing
from concurrent.futures import ThreadPoolExecutor # For background file saving
# Import utils
from video_processing.utils import get_video_stats, check_blur, save_frame_worker
from video_processing.tiling import generate_tiles

#Logic to find the config relative to the Project Root
#By doing this, the code works on any computer because it doesn't care about the folders above the project(our project is at 2026_T1 folder level)
BASE_DIR = os.getcwd()
#join() is used to build a path to json config file 
CONFIG_PATH = os.path.join(BASE_DIR, "shared", "config", "video_processing_config.json")

#max_workers 4 reserved for writing extracted frame to folder where we want to save 
executor = ThreadPoolExecutor(max_workers=4)

#Important paths and parameters are stored in this config file which can be updated if needed
#We load and read that config file
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def runtime_preprocessing(frame, config):
    """
    Runtime preprocessing stage for the T2 pipeline.

    Current implementation:
        - Configurable tiling.

    Future additions:
        - Use video-level quality thresholds
          calculated during video analysis.
        - Conditional CLAHE.
        - Configurable tile overlap.
        - Optional letterboxing.

    The frame is currently returned unchanged unless
    tiling is enabled.
    """

    # --------------------------------------------------------
    # FUTURE T2: QUALITY-BASED PREPROCESSING
    # --------------------------------------------------------
    #
    # Video-level quality statistics will be calculated
    # during the initial video analysis stage.
    #
    # For example:
    #
    #     blur threshold
    #     contrast threshold
    #     brightness threshold
    #     sharpness threshold
    #
    # The current frame can later be compared against
    # these thresholds here to determine whether
    # preprocessing such as CLAHE is required.
    #
    # --------------------------------------------------------


    # For now, keep the original frame unchanged.
    processed_frame = frame


    # --------------------------------------------------------
    # CURRENT: CONFIGURABLE TILING
    # --------------------------------------------------------

    tiling_enabled = config.get(
        "enable_tiling",
        False
    )


    tiles = []
    tile_metadata = []


    if tiling_enabled:

        tile_rows = config.get(
            "tile_rows",
            2
        )

        tile_columns = config.get(
            "tile_columns",
            2
        )


        # ----------------------------------------------------
        # FUTURE T2: TILE OVERLAP
        # ----------------------------------------------------
        #
        # Later we will read something such as:
        #
        # tile_overlap = config.get(
        #     "tile_overlap",
        #     0
        # )
        #
        # and pass it to generate_tiles().
        #
        # Current tiling is non-overlapping.
        #
        # ----------------------------------------------------

        tiles, tile_metadata = generate_tiles(
            processed_frame,
            rows=tile_rows,
            cols=tile_columns
        )


    # --------------------------------------------------------
    # FUTURE T2: LETTERBOXING
    # --------------------------------------------------------
    #
    # Later:
    #
    # if config.get("enable_letterbox", False):
    #     ...
    #
    # Letterboxing will prepare the frame/tile for the
    # downstream detection model while preserving aspect ratio.
    #
    # The target size should come from the detector's
    # input requirements.
    #
    # --------------------------------------------------------


    return (
        processed_frame,
        tiles,
        tile_metadata
    )

def process_video(video_id: str, video_path: str):
    """
    video_path: Expected as 'data/raw/filename.mp4' (relative to Root (2026_T1))
    """
    config = load_config()
    
    #normpath will clean up any accidental double slashes (like data//raw) to ensure the OS can find the file
    #contains path where input video is present
    full_input_path = os.path.normpath(os.path.join(BASE_DIR, video_path))
    
    # Existing video quality analysis.
    #
    # CURRENT:
    #   Calculates the dynamic blur threshold.
    #
    # FUTURE T2:
    #   Extend this analysis stage to also calculate
    #   video-level contrast, brightness and sharpness
    #   statistics and derive their thresholds.
    #
    # This keeps all video-level quality analysis
    # in one place rather than creating a separate
    # quality-analysis pass.

    # Establish the 'Sharpness Floor' for this specific crowd footage
    print(f"Analyzing crowd video quality for {video_id}...")
    # If variance is less then threshold blurry image else sharp image
    dynamic_threshold = get_video_stats(full_input_path, config["sample_rate"])
    print(f"Calculated Crowd Quality Threshold: {dynamic_threshold:.2f}")

    
    # ------------------------------------------------------------
    # Frame and Tile Paths
    # ------------------------------------------------------------
    #contains path where output frames will be stored
    output_dir = os.path.join(BASE_DIR, config["extracted_frames_dir"])
    #It creates folder where output frames will be stored if only folder is already not created
    os.makedirs(output_dir, exist_ok=True)

    #Contains path where generated tiles will be stored.
    tiled_output_dir = os.path.join(
                        BASE_DIR,
                        "video_processing",
                        "data",
                        "tiled_frames"
                    )
    os.makedirs(
                        tiled_output_dir,
                        exist_ok=True
                    )

    
    
    #opens video stream
    cap = cv2.VideoCapture(full_input_path)
    #if video file is corrupted or path is wrong we return an error
    if not cap.isOpened():
        return {"error": f"Could not open video at {full_input_path}"}

    #We store frames per second of video, if opencv cant find out we fallback to 30fps to avoid division by zero error later
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    
    frames_metadata = []
    save_futures = []
    count = 0
    extracted_count = 1 

    print(f"--- Processing Video: {video_path} ---")

    try:
        while True:
            #we read video frame by frame
            ret, frame = cap.read()
            #if no frames left, we get out of loop
            if not ret: break
            
            #Frame Sampling (We take snapshot every 30 frames, instead of taking snapshot of all frames) 
            if count % config["sample_rate"] == 0:
                score, is_sharp = check_blur(frame, dynamic_threshold)
                
                # If the camera is panning or shaking(blurry frame), check the next few frames.
                # Crowd faces are unrecognizable in motion blur.
                search_count = 0
                while not is_sharp and search_count < 8: # Slightly longer window for crowd stabilization
                    ret, frame = cap.read()
                    if not ret: break
                    count += 1
                    search_count += 1
                    score, is_sharp = check_blur(frame, dynamic_threshold)
                
                # ------------------------------------------------------------
                # T2 RUNTIME PREPROCESSING
                # ------------------------------------------------------------
                #
                # Current:
                #   - Tiling
                #
                # Future:
                #   - Compare current-frame quality against
                #     video-level thresholds.
                #   - Conditional CLAHE.
                #   - Tile overlap.
                #   - Letterboxing.
                #
                processed_frame, tiles, tile_metadata = runtime_preprocessing(
                    frame,
                    config
                )

                #frame naming for maintaining frame order
                fname = f"frame_{extracted_count:04d}.jpg"
                save_path = os.path.join(output_dir, fname)
                
                save_futures.append(executor.submit(save_frame_worker, save_path, processed_frame))

               
                # ------------------------------------------------------------
                # CURRENT T2: SAVE TILES
                # ------------------------------------------------------------

                if config.get(
                    "enable_tiling",
                    False
                ):

                    for tile_index, tile in enumerate(
                        tiles
                    ):

                        tile_filename = (
                            f"frame_"
                            f"{extracted_count:04d}_"
                            f"tile_"
                            f"{tile_index + 1}.jpg"
                        )


                        tile_path = os.path.join(
                            tiled_output_dir,
                            tile_filename
                        )


                        # Use the existing background saving worker
                        # for tiles as well as normal frames.

                        save_futures.append(
                            executor.submit(
                                save_frame_worker,
                                tile_path,
                                tile
                            )
                        )


                        # Add the tile's output path to its metadata.

                        tile_metadata[tile_index]["tile_path"] = (
                            f"video_processing/data/"
                            f"tiled_frames/"
                            f"{tile_filename}"
                        )



                #Match the 'DetectionFrame' schema in shared/models.py
                frames_metadata.append({
                    "frame_id": extracted_count,
                    #this will tell us at what time the frame is present in video
                    "timestamp": round(count / fps, 2),
                    "frame_path": f"{config['extracted_frames_dir']}/{fname}",
                    # Attach tile metadata to this frame.
                    # If tiling is disabled, this will simply be an empty list.
                    "tiles": tile_metadata
                })
                    


                extracted_count += 1
            count += 1
    finally:
        #This "closes" the video file. If we don't do this, the computer might keep the file "locked," and we won't be able to delete or move it until we restart the PC
        cap.release()

    for future in save_futures:
        future.result()

    #Return the dictionary for the Service Layer to use
    return {
        "video_id": video_id,
        "video_path": video_path,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frames": frames_metadata
    }

if __name__ == "__main__":
    #Run this from the Project Root (2026_T1)
    #python -m video_processing.main
    test_res = process_video("match_01", "data/raw/match_01.mp4")
    if "error" in test_res:
        print(test_res["error"])

    else:
        print(
            f"Successfully processed "
            f"{len(test_res['frames'])} frames."
        )