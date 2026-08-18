import cv2


def generate_tiles(frame, rows=2, cols=2):
    """
    Split an image into equal-sized tiles.

    Parameters
    ----------
    frame : numpy.ndarray
        Input image.
    rows : int
        Number of tile rows.
    cols : int
        Number of tile columns.

    Returns
    -------
    tiles : list
        List of cropped tile images.
    metadata : list
        Position information for each tile.
    """

    # Validate configuration
    if rows <= 0:
        raise ValueError("rows must be greater than 0")

    if cols <= 0:
        raise ValueError("cols must be greater than 0")

    # Get original frame dimensions
    frame_height, frame_width = frame.shape[:2]


    # Calculate tile dimensions.
    #
    # Example:
    #
    # Frame = 1920 x 1080
    # rows = 2
    # cols = 2
    #
    # tile_width  = 1920 // 2 = 960
    # tile_height = 1080 // 2 = 540
    tile_width = frame_width // cols
    tile_height = frame_height // rows

    tiles = []
    metadata = []

    tile_id = 1

    # Generate each tile
    for row in range(rows):
        for col in range(cols):

            #Calculate start position for every tile
            #For tile 1, x = 0, y = 1080
            x = col * tile_width
            y = row * tile_height

            # Crop the tile from the original frame.
            #
            # frame[y:y + tile_height,
            #       x:x + tile_width]
            tile = frame[
                #For tile 1,  0:1080, 0:1920
                y:y + tile_height,
                x:x + tile_width
            ]

            # Store the actual tile
            tiles.append(tile)

            # Store information about where the tile
            # came from in the original frame.
            metadata.append({
                "tile_id": tile_id,
                "row": row,
                "column": col,
                "x": x,
                "y": y,
                "width": tile_width,
                "height": tile_height
            })

            tile_id += 1

    return tiles, metadata