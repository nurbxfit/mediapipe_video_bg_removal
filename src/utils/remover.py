import os
import mediapipe as mp 
import cv2
import os
from utils.video import get_video_info, read_video, create_output_process
import math
import numpy as np
from tqdm import tqdm
from utils.bg_overlay import apply_bg


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
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True,
    )

    with ImageSegmenter.create_from_options(options) as segmenter:
        width, height = get_video_info(video_path)
        print(f"WIDTH:{width}, HEIGHT: {height}")

        ffmpeg_process = create_output_process('output',width,height)

        frame_count = 0
        frames = read_video(video_path, width, height)
        with tqdm(desc='Processing video', unit=' frames') as pbar:
            for frame in frames:
                # Process the frame
                removed_bg = remove_bg(frame, segmenter)
                added_bg = apply_bg(removed_bg)

                # Convert removed_bg to np.uint8
                output_uint8 = (added_bg * 255).astype(np.uint8)
                
                # write it to file
                ffmpeg_process.stdin.write(output_uint8.tobytes())
                frame_count += 1 # just for tqdm
                pbar.update(1)



                show_debug(added_bg)
                # show_debug(removed_bg,width,height)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cv2.destroyAllWindows()
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()


def remove_bg(frame, selfie_segmentation, replacement_color=(0, 0, 0)):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))
    segmentation_result = selfie_segmentation.segment(image)
    category_mask = segmentation_result.category_mask

    # Create an alpha channel based on the category mask
    alpha_channel = (category_mask.numpy_view() > 0.1).astype(np.uint8) * 255

    # Create a mask where person's region is white and background is black
    mask = (category_mask.numpy_view() > 0.1).astype(np.uint8) * 255

    # Invert the mask to make the background white and person's region black
    inverted_mask = cv2.bitwise_not(mask)

    # Convert the frame to RGBA format
    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

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