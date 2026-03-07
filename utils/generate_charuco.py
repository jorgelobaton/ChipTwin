import cv2
import numpy as np
from PIL import Image  # Requires: pip install pillow
import json
import os

# Compute paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
calibration_path = os.path.join(script_dir, "..", "cams", "calibration.json")

with open(calibration_path, "r") as f:
    calib_data = json.load(f)


def generate_exact_charuco():
    # 1. Configuration
    squares_x = calib_data["squares_x"]
    squares_y = calib_data["squares_y"]
    square_length_mm = calib_data["square_length_mm"]
    margin_mm = calib_data["margin_mm"]
    target_dpi = calib_data["target_dpi"]
    
    # 2. Calculate Exact Pixel Dimensions
    # Formula: pixels = (mm / 25.4) * dpi
    square_px = int((square_length_mm / 25.4) * target_dpi)
    margin_px = int((margin_mm / 25.4) * target_dpi)
    
    # Total canvas size
    active_width_px = squares_x * square_px
    active_height_px = squares_y * square_px
    total_width_px = active_width_px + (2 * margin_px)
    total_height_px = active_height_px + (2 * margin_px)
    
    print(f"Generating {squares_x}x{squares_y} board @ {target_dpi} DPI")
    print(f"Square size: {square_px}px ({square_length_mm}mm)")
    print(f"Total Image Size: {total_width_px}x{total_height_px} pixels")

    # 3. Create Board Object (Physical units here only affect calibration logic)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        squareLength=square_length_mm/1000, 
        markerLength=calib_data["markerLength"],
        dictionary=dictionary
    )
    
    # 4. Generate Image using calculated pixels
    img = board.generateImage(
        (total_width_px, total_height_px), 
        marginSize=margin_px, 
        borderBits=1
    )
    
    # 5. Save with DPI Metadata
    # cv2.imwrite() discards DPI. We use Pillow to save it correctly.
    output_filename = f"charuco_{squares_x}x{squares_y}_{target_dpi}dpi.png"
    pil_image = Image.fromarray(img)
    pil_image.save(output_filename, dpi=(target_dpi, target_dpi))
    
    print(f"Saved to '{output_filename}'.")
    print("PRINTING INSTRUCTIONS: Open in system viewer and print at '100% Scale'.")

if __name__ == "__main__":
    generate_exact_charuco()
