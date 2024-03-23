import mediapipe as mp
import os

MODEL_PATH=os.path.join('models/selfie_segmenter.tflite')

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

def get_segmented_frame(image, segmenter,timestamp = 0):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = segmenter.segment(mp_image)
    mask = result.category_mask
    return mask.numpy_view()