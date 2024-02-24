import os
import mediapipe as mp
import numpy as np
from PIL import Image
from utils import manipulator
import cv2
import math

MODEL_PATH=os.path.join('models/selfie_segmenter.tflite')
DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 920

def process_frame(image,segmenter):
    # how do I check if image passed is actually a numpy array ?
    # if  image.shape ? help me!

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = segmenter.segment(mp_image)
    mask = result.category_mask
    mask = mask.numpy_view()

    foreground_image = manipulator.remove_bg_v2(image,mask)

    new_bg = load_local_image('overlays/xp-bg.jpg')
    applied_bg = manipulator.blend_bg(foreground_image,new_bg,mask)
    return applied_bg
    # return removed_bg


def init_segmenter(model_path=MODEL_PATH, running_mode = 'video'):
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    RunningMode = VisionRunningMode.VIDEO if running_mode == 'video' else VisionRunningMode.IMAGE

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode,
        output_category_mask=True,
    )

    return ImageSegmenter.create_from_options(options)


def main():
    image = load_local_image('input/example.jpg',)
    with init_segmenter(MODEL_PATH,'image') as segmenter:
        processed_frame = process_frame(image,segmenter)
        show_debug(processed_frame)

    # # cv2.destroyAllWindows()
    # cv2.waitKey(0)


def load_local_image(path):
    image_path = os.path.join(path)
    image = Image.open(image_path)
    image = np.array(image)
    return image

def show_debug(frame,width = DESIRED_WIDTH, height = DESIRED_HEIGHT):
    h, w = frame.shape[:2]
    if h < w:
        img = cv2.resize(frame, (width, math.floor(h/(w/width))))
    else:
        img = cv2.resize(frame, (math.floor(w/(h/height)), height))
    cv2.imshow('Preview',img)
    # Press any key to close the window
    cv2.waitKey(0)

    # Explicitly close the window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()