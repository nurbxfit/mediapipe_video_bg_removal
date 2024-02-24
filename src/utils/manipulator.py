import cv2
import numpy as np

# Note: images params here are the numpy array
#
def blend_bg(fg,bg,mask):
    
    # Ensure alpha channel for fg
    fg_mask = (mask < 0.3).astype(np.uint8) * 255
    bg_mask = (mask > 0.3).astype(np.uint8) * 255
    bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))
   
    bg = cv2.cvtColor(bg,cv2.COLOR_BGR2RGBA)
    # bg_mask = cv2.bitwise_not(bg)

    cutof_with_bg_as_mask = cv2.bitwise_and(bg,bg,mask=fg_mask)
    bg_with_cutof_black = cv2.bitwise_and(bg,bg,mask=bg_mask)
    # cv2.imshow('fg_mask',fg_mask)
    # cv2.imshow('bg_mask',bg_mask)
    # cv2.imshow('bg_with_cutof',bg_with_cutof_black)
    # cv2.imshow('cutof_with_bg_as_mask',cutof_with_bg_as_mask)

    # fg_mask_tiled = np.tile(fg_mask[:,:,None],(1,1,4))
    print(f"mask:{fg_mask.shape}")
    print(f"fg:{fg.shape}")

    blended_fg = cv2.addWeighted(fg,1,cutof_with_bg_as_mask,0.2,0)
    blended_fg_bg = blended_fg + bg_with_cutof_black
    # gaussian blur
    output_gaus = cv2.GaussianBlur(blended_fg_bg,(5,5),0)
    output_gaus = cv2.normalize(output_gaus, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    blended = output_gaus
    # bg_with_fg_cutout= cv2.bitwise_and(bg_rgba,bg_rgba,mask=fg_mask)

    return (blended).astype(np.uint8)



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


def remove_bg_v2(frame,category_mask,replacement_color=(0,0,0)):
    # Create an alpha channel from the category mask
    # Lowering the threshold (e.g., 0.05): This includes more pixels in the foreground mask, 
    # potentially capturing finer details and potentially including some background noise.
    # Raising the threshold (e.g., 0.15): This excludes more pixels from the foreground mask, 
    # resulting in a "tighter" segmentation that might miss some true foreground pixels but exclude more background noise.
    fg_mask_threshold = 0.3
    fg_mask = (category_mask < fg_mask_threshold).astype(np.uint8) * 255
    # bg_mask = (category_mask.numpy_view() > 0.1).astype(np.uint8) * 255

    frame_rgba = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA) # we use BGR to RGBA because the result of mp.Image().segment is BGR

    bg_transparent = np.full_like(frame_rgba, replacement_color + (255,),dtype=np.uint8)
    # bg_normal = np.full_like(frame_rgba,frame_rgba,dtype=np.uint8)
    
    # I want to apply some kind of blending to smooth up the foreground image edges
    # but I am not sure how to do it.
    # Apply feathering to the foreground mask
    # kernel_size = 3  # Adjust kernel size for desired feathering strength
    # fg_mask_blurred = cv2.GaussianBlur(fg_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    # fg_mask_blurred = cv2.normalize(fg_mask_blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)  # Normalize to 0-255 range

    #cut the segmented fg using the mask, then we left with only the foreground image
    # Isolate the foreground with alpha channel using the feathered mask
    # fg_mask_blurred = fg_mask_blurred.astype(np.uint8)
    # fg_with_removed_bg = cv2.bitwise_and(frame_rgba, frame_rgba, mask=fg_mask[..., None])

    fg_with_removed_bg = cv2.bitwise_and(frame_rgba, frame_rgba,mask=fg_mask) 
    bg_mask = cv2.bitwise_not(fg_mask)
    # then we transparent transparent background
    # we cut holes in it matching the foregound image
    bg_transparent_with_removed_fg = cv2.bitwise_and(bg_transparent, bg_transparent, mask=bg_mask)

    # then we stich it togerther forming a fg image + transparent bg
    output = fg_with_removed_bg + bg_transparent_with_removed_fg

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