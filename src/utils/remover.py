import os
import mediapipe as mp 
import cv2
import os
from utils.video import get_video_info, read_video, create_output_process
import math
import numpy as np
from tqdm import tqdm
from utils.bg_overlay import apply_bg, apply_fg
from queue import Queue
import threading

MODEL_PATH=os.path.join('models/selfie_segmenter.tflite')
DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 920
BG_COLOR = (0,0,0) # black

def process_video(video_path,model_path=MODEL_PATH):
    print(f"video_path:{video_path}")
    print(f"model_path:{model_path}")

    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        # running_mode=VisionRunningMode.IMAGE,
        running_mode=VisionRunningMode.VIDEO,
        output_category_mask=True,
    )


    with ImageSegmenter.create_from_options(options) as segmenter:
        width, height = get_video_info(video_path)
        print(f"WIDTH:{width}, HEIGHT: {height}")

        ffmpeg_process = create_output_process('output', width, height)
        # ffmpeg_process = create_output_process('output',width,height)

        # frame_count = 0
        caps_queue = Queue(maxsize=128)
        processed_queue = Queue(maxsize=128)
        final_queue = Queue(maxsize=128)

        # Here we create threads for reading,process and writing
        read_thread = threading.Thread(target=read_frames, args=(video_path, width, height, caps_queue))
        process_thread = threading.Thread(target=process_frames, args=(segmenter, caps_queue, processed_queue))
        apply_thread = threading.Thread(target=apply_frames, args=(processed_queue, final_queue))
        write_thread = threading.Thread(target=write_frames, args=(final_queue, ffmpeg_process))

        # Start the threads
        read_thread.start()
        process_thread.start()
        apply_thread.start()
        write_thread.start()

        # Wait for all threads to finish
        read_thread.join()
        process_thread.join()
        apply_thread.join()
        write_thread.join()

        cv2.destroyAllWindows()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait() 


def read_frames(video_path, width, height, caps_queue, target_frame_rate = 24):
    caps = read_video(video_path, width, height,target_frame_rate)
    # with tqdm(desc='Reading video', unit=' frames') as pbar:
    for cap in caps:
        frame, timestamp = cap
        if not timestamp:
            break
        caps_queue.put(cap)
        # pbar.update(1)
    caps_queue.put(None)

def process_frames(segmenter, caps_queue, processed_queue):
    while True:
        cap  = caps_queue.get()
        if cap is None:
            break
        frame, timestamp = cap
        removed_bg = remove_bg(frame,segmenter,timestamp)
        
        processed_queue.put((removed_bg,timestamp))
    processed_queue.put(None)

def apply_frames(processed_queue,final_queue):
    while True:
        processed_frame = processed_queue.get()
        if processed_frame is None:
            break
        frame, timestamp = processed_frame
        added_bg = apply_bg(frame)
        final_queue.put((added_bg,timestamp))
    final_queue.put(None)

def write_frames(final_queue, ffmpeg_process):
    with tqdm(desc='Saving video', unit=' frames') as pbar:
        while True: 
            processed_frame = final_queue.get()
            if processed_frame is None:
                break
            frame, timestamp = processed_frame
            ffmpeg_process.stdin.write(frame)
            pbar.update(1)

def remove_bg(frame, selfie_segmentation, timestamp, replacement_color=(0, 0, 0)):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))

    # segmentation_result = selfie_segmentation.segment(image)
    segmentation_result = selfie_segmentation.segment_for_video(image,timestamp)
    category_mask = segmentation_result.category_mask

    # Create an alpha channel based on the category mask
    alpha_channel = (category_mask.numpy_view() > 0.1).astype(np.uint8) * 255

    # Create a mask where person's region is white and background is black
    mask = (category_mask.numpy_view() > 0.1).astype(np.uint8) * 255

    # Invert the mask to make the background white and person's region black
    inverted_mask = cv2.bitwise_not(mask)

    # Convert the frame to RGBA format
    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    # Set the background to the replacement color 
    background = np.full_like(frame_bgra, replacement_color + (255,), dtype=np.uint8)

    # Combine the frame and the background using the masks and alpha channel
    output_image = cv2.bitwise_and(frame_bgra, frame_bgra, mask=inverted_mask)
    output_image += cv2.bitwise_and(background, background, mask=mask)
    output_image[:, :, 3] = alpha_channel

    return output_image





def show_debug(frame,width = DESIRED_WIDTH, height = DESIRED_HEIGHT):

    h, w = frame.shape[:2]
    if h < w:
        img = cv2.resize(frame, (width, math.floor(h/(w/width))))
    else:
        img = cv2.resize(frame, (math.floor(w/(h/height)), height))
    cv2.imshow('Preview',img)