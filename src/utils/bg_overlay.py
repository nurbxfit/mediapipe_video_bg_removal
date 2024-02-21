import cv2
import os
import mediapipe as mp
import numpy as np

DEFAULT_BG_PATH = os.path.join("overlays","xp-bg.jpg")

def apply_bg(fg_image, bg_image_path = DEFAULT_BG_PATH):
    # Read the background image
    background = cv2.imread(bg_image_path)

    # Resize the background to match the dimensions of the input frame
    background = cv2.resize(background, (fg_image.shape[1], fg_image.shape[0]))

    # Create a mask for the foreground image (with removed bg)
    mask = fg_image[:, :, 3]

    # Invert the mask to select the region where the new background will be applied
    inverted_mask = cv2.bitwise_not(mask)

    # Extract the region from the original frame without the removed background
    roi = fg_image[:, :, :3]

    # Extract the region from the new background
    bg_roi = background[:, :, :3]

    # Apply the new background to the selected region
    result = cv2.bitwise_and(roi, roi, mask=inverted_mask)
    result += cv2.bitwise_and(bg_roi, bg_roi, mask=mask)

    # Create an RGBA image with the new background
    output_image = np.zeros_like(fg_image)
    output_image[:, :, :3] = result
    output_image[:, :, 3] = mask


    return output_image