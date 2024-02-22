import cv2
import os
import numpy as np

DEFAULT_BG_PATH = os.path.join("overlays","main-bg.png")
OVERLAY_KETUPAT_RIGHT = os.path.join('overlays','ketupat-2-left.png')

def apply_bg(fg_image, bg_image_path = DEFAULT_BG_PATH):
    # Read the background image
    # background = cv2.imread(bg_image_path,cv2.IMREAD_UNCHANGED)
    background = cv2.imread(bg_image_path)

    # Resize the background to match the dimensions of the input frame
    background = cv2.resize(background, (fg_image.shape[1], fg_image.shape[0]))
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGBA)

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


def apply_fg(background, overlay_path=OVERLAY_KETUPAT_RIGHT, position=[0, 0], resize=[0.3, 0.3], alignment='left'):
    resize_scale_x, resize_scale_y = resize
    position_x, position_y = position

    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    overlay = cv2.resize(overlay, (0, 0), fx=resize_scale_x, fy=resize_scale_y)

    overlay_height, overlay_width, overlay_c = overlay.shape
    background_height, background_width, background_c = background.shape

    if alignment == 'right':
        position_x = background_width - overlay_width - position_x
    elif alignment == 'center':
        position_x = (background_width - overlay_width) // 2 + position_x

    position_x2, position_y2 = min(position_x + overlay_width, background_width), min(
        position_y + overlay_height, background_height)

    overlay_x = 0 if position_x >= 0 else -position_x
    overlay_y = 0 if position_y >= 0 else -position_y

    overlay_width = position_x2 - position_x
    overlay_height = position_y2 - position_y

    if overlay_width <= 0 or overlay_height <= 0 or overlay_width > background_width or overlay_height > background_height:
        return background

    mask = overlay[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width, 3] / 255.0
    inverse_mask = 1.0 - mask

    image_rgb = overlay[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width, 0:3]

    for c in range(0, 3):
        background[position_y:position_y2, position_x:position_x2, c] = background[
            position_y:position_y2, position_x:position_x2, c] * inverse_mask + image_rgb[:, :, c] * mask

    return background

