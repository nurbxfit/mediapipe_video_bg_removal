import cv2
import numpy as np

# Note: images params here are the numpy array
#
def blend_bg(fg,bg):
    
    # Ensure alpha channel for fg
    fg = create_alpha(fg).astype(float)

    # Ensure alpha channel for bg
    bg = create_alpha(bg).astype(float)

    bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))

    blended = fg * 0.5 + bg * 0.5

    return (blended * 255).astype(np.uint8)


def create_alpha(image):
    if image.shape[2] == 3:
        # image_alpha = np.full(image.shape[:2],255,dtype=np.uint8)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGBA)
        # image = np.concatenate([image,image_alpha],axis=-1)

    return image

# note: frame is numpy  array read from buffer 
#  frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
# category_mask it the result of ( mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame)).segmentation_result.category_mask)
def remove_bg(frame,category_mask,replacement_color=(0,0,0)):
    # Create an alpha channel of the category mask
    alpha_channel = (category_mask.numpy_view() > 0.1).astype(np.uint8) * 255

    # create inverted alpha 
    inverted_alpha = cv2.bitwise_not(alpha_channel)

    # create an alpha channel for our frame (RGBA)
    frame_rgba = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA) # we use BGR to RGBA because the result of mp.Image().segment is BGR

    # we copy the frame array, but fill it with solid color default (0,0,0) black
    # + (255,) means adding alpha channel to it , control the transparency
    background = np.full_like(frame_rgba,replacement_color + (255,), dtype=np.uint8)

    # combine the frame and background using the mask and alpha channel
    output = cv2.bitwise_and(frame_rgba, frame_rgba, mask=inverted_alpha)
    output += cv2.bitwise_and(background,background, mask=alpha_channel)
    output[:,:,3] = alpha_channel

    # convert the end output to RGBA

    return cv2.cvtColor(output,cv2.COLOR_BGRA2RGBA)

# Note: fg, and bg are numpy array
def apply_fg(bg,fg, positon=(0,0), resize=(0,0), alignment='left'):
    resize_scale_x , resize_scale_y = resize
    position_x, position_y = positon

    fg = cv2.resize(fg,(0,0),fx=resize_scale_x, fy=resize_scale_y)

    fg_height , fg_width , fg_c = fg.shape
    bg_height, bg_width, bg_c = bg.shape

    if alignment == 'right':
        position_x = bg_width - fg_width - position_x
    elif alignment == 'center':
        position_x = (bg_width - fg_width) // 2 + position_x

    position_x2, position_y2 =  min(position_x + fg_width, bg_width), min(
        position_y + fg_height, bg_height)
    
    fg_x = 0 if position_x >= 0 else -position_x
    fg_y = 0 if position_y >= 0 else -position_y

    fg_width = position_x2 - position_x
    fg_height = position_y2 - position_y

    if fg_width <= 0 or fg_height <= 0 or fg_width > bg_width or fg_height > bg_height:
        return bg

    mask = fg[fg_y:fg_y + fg_height, fg_x:fg_x + fg_width, 3] / 255.0
    inverse_mask = 1.0 - mask

    image_rgb = fg[fg_y:fg_y + fg_height, fg_x:fg_x + fg_width, 0:3]

    for c in range(0, 3):
        bg[position_y:position_y2, position_x:position_x2, c] = bg[
            position_y:position_y2, position_x:position_x2, c] * inverse_mask + image_rgb[:, :, c] * mask

    return bg