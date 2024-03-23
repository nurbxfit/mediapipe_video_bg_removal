import cv2 
import numpy as np

def remove_background(frame,mask, replacement_color=(0,0,0)):
    # adjust the color 
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # create a mask where person is white, bg is black
    invert_mask = cv2.bitwise_not(mask)

    # expand the mask a little bit
    erode_kernel =  np.ones((7,7),np.uint8)
    # dilate_kernel =  np.ones((17,17),np.uint8)
    invert_mask = cv2.erode(invert_mask,kernel=erode_kernel) # shrink
    # invert_mask = cv2.dilate(invert_mask,kernel=dilate_kernel) # expand

    # blur the mask for smooth out the edges
    person_mask = cv2.blur(invert_mask,(21,21))
    person_mask = cv2.normalize(person_mask,None,alpha=0,beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    person_maskf = (person_mask/255).astype(np.float64)
    person_maskf = cv2.merge([person_maskf,person_maskf,person_maskf]) # create 3 channel mask
    

    frame_without_bg = (person_maskf * frame).astype(np.uint8)

    final = cv2.cvtColor(frame_without_bg,cv2.COLOR_BGR2RGBA)
    return final