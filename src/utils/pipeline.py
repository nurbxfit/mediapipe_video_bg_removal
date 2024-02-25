import os
import mediapipe as mp
import numpy as np
from PIL import Image
from utils import manipulator, video, segmentation
import cv2
import math
import concurrent.futures as cf
from tqdm import tqdm
import time


DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 920

timestamp = 0
def process_frame(image,segmenter):
    # how do I check if image passed is actually a numpy array ?
    # if  image.shape ? help me!
    global timestamp
    timestamp +=1
    result = segmentation.get_segmented_frame(image,segmenter)
    mask = result.category_mask
    mask = mask.numpy_view()

    foreground_image = manipulator.remove_bg_v2(image,mask)

    # new_bg = load_local_image('overlays/xp-bg.jpg')
    # applied_bg = manipulator.blend_bg(foreground_image,new_bg,mask)
    # return applied_bg, timestamp
    return foreground_image , timestamp

def process_frames(frames,segmenter):
    
    processed_frames = []
    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for frame,_ in frames:
            future = executor.submit(process_frame,frame,segmenter)
            futures.append(future)
        
        with tqdm(desc="Processing video",unit=" frames") as pbar:
            for future in cf.as_completed(futures):
                result = future.result()
                processed_frames.append(result)
                pbar.update(1)
    # sort the array, before write it
    processed_frames.sort(key=lambda x: x[1])
    return processed_frames

def write_frames(frames,width, height,dest_path = 'output'):
    ffmpeg_process = video.create_output_process(dest_path, width, height)
    with tqdm(desc="Saving video", unit=" frames") as pbar:
        for frame, _ in frames:
            if frame is None:
                pass
            else:
                ffmpeg_process.stdin.write(frame)
                pbar.update(1)

    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    



def main():
    start_time = time.time()
    run_video_processing()
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = "{:.2f} seconds".format(elapsed_time)
    print("Program execution time:", formatted_time)



def run_video_processing():
    input_path = 'input/video-input.mp4'
    # MODEL_PATH = 'models/selfie_multiclass_256x256.tflite'
    MODEL_PATH = 'models/selfie_segmenter.tflite'
    # image = load_local_image('input/example.jpg',)
    cap = load_local_video(input_path)
    width,height = video.get_video_info(input_path)
    with segmentation.init_segmenter(MODEL_PATH,'image') as segmenter:
        # processed_frame = process_frame(image,segmenter)
        # show_debug(processed_frame)
        processed_frames = process_frames(cap,segmenter)
        write_frames(processed_frames,width,height)
    
    # # cv2.destroyAllWindows()
    # cv2.waitKey(0)

def load_local_video(path):
    cap = video.read_video(path)
    return cap

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