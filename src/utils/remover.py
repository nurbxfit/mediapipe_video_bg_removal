import os
import mediapipe as mp 
import cv2
import os
from utils.video import get_video_info, read_video, create_output_process
import math
import numpy as np
from tqdm import tqdm
import ffmpeg


MODEL_PATH=os.path.join('models/selfie_segmenter.tflite')
DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 920
BG_COLOR = (0,0,0) # black

def remove_bg(video_path,model_path=MODEL_PATH):
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
        # ffmpeg_process = (
        #     ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgba', s='{}x{}'.format(width, height))
        #     .output('output.mp4', vcodec='libx264', r=24)
        #     .overwrite_output()
        #     .run_async(pipe_stdin=True)
        # )

        ffmpeg_process = create_output_process('output',width,height)

        frame_count = 0
        frames = read_video(video_path, width, height)
        with tqdm(desc='Processing video', unit=' frames') as pbar:
            for frame in frames:
                # Process the frame
                output_image = process_video(frame, segmenter)

                # Convert output_image to np.uint8
                output_image_uint8 = (output_image * 255).astype(np.uint8)
                
                # write it to file
                ffmpeg_process.stdin.write(output_image_uint8.tobytes())
                frame_count += 1 # just for tqdm
                pbar.update(1)

                # show_debug(output_image,width,height)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cv2.destroyAllWindows()
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()


def process_video(frame, selfie_segmentation):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))
    segmentation_result = selfie_segmentation.segment(image)
    category_mask = segmentation_result.category_mask

    # Convert the BGR image back to RGB
    # image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
    image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)


    # Make background transparent
    bg_image = np.zeros((image_data.shape[0], image_data.shape[1], 4), dtype=np.uint8)
    # bg_image[:, :, :3] = BG_COLOR  # Set RGB values for the background
    # bg_image[:, :, 3] = 0  # Set alpha channel to 0 for the background

    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) <= 0.1
    # output_image = np.where(condition, image_data, bg_image)
    # Create an RGBA image by combining the original image and the transparent background
    output_image = np.zeros_like(bg_image)
    output_image[:, :, :3] = np.where(condition, image_data, bg_image[:, :, :3])
    output_image[:, :, 3] = np.where(condition[:, :, 0], 0, bg_image[:, :, 3])

    return output_image



def show_debug(frame,width = DESIRED_WIDTH, height = DESIRED_HEIGHT):

    h, w = frame.shape[:2]
    if h < w:
        img = cv2.resize(frame, (width, math.floor(h/(w/width))))
    else:
        img = cv2.resize(frame, (math.floor(w/(h/height)), height))
    cv2.imshow('Preview',img)