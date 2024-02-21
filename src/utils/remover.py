import os
import mediapipe as mp 
import cv2
import os
from utils.video import get_video_info, read_video, create_output_process
import math
import numpy as np
from tqdm import tqdm
from utils.bg_overlay import apply_bg, apply_fg
import concurrent.futures as cf
from mediapipe.python._framework_bindings.timestamp import Timestamp

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
    VisionRunningModeOption = VisionRunningMode.VIDEO
    # VisionRunningModeOption = VisionRunningMode.IMAGE

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningModeOption,
        output_category_mask=True,
    )

    with ImageSegmenter.create_from_options(options) as segmenter:
        width, height, frame_rate, duration = get_video_info(video_path)

        ffmpeg_process = create_output_process('output',width,height)

        target_frame_rate = 24
        frames = read_video(video_path, width, height, target_frame_rate)
        processed_frames = []
        with cf.ThreadPoolExecutor() as executor:
            futures = []
            for frame, timestamp in frames:
                future = executor.submit(process_frames,frame, segmenter, timestamp)
                futures.append(future)
            with tqdm(desc='Processing video', unit=' frames') as pbar:
                for future in cf.as_completed(futures):
                    processed_frame = future.result()
                    processed_frames.append(processed_frame)
                    pbar.update(1)

            

        with tqdm(desc="Saving video", unit=" frames") as pbar:
            for frame in processed_frames:
                ffmpeg_process.stdin.write(frame)
                pbar.update(1)

        cv2.destroyAllWindows()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
                

def process_frames(frame, segmenter, timestamp):
    removed_bg = remove_bg(frame, segmenter, timestamp)
    return apply_bg(removed_bg)

def remove_bg(frame, selfie_segmentation, timestamp, replacement_color=(0, 0, 0), dilate_kernel_size=5, blur_kernel_size=5):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))

    segmentation_result = selfie_segmentation.segment_for_video(image,timestamp)
    # segmentation_result = selfie_segmentation.segment(image)
    category_mask = segmentation_result.category_mask

    # Create an alpha channel based on the category mask
    # alpha_channel = (category_mask.numpy_view() > 0.3).astype(np.uint8) * 255

    # Create a mask where person's region is white and background is black
    mask = (category_mask.numpy_view() > 0.3).astype(np.uint8) * 255
    alpha_channel = mask

    # Perform dilation and erosion to smooth out the edges of the mask
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Apply Gaussian blur to the mask
    mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

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