import os
from utils import segmentation, remover, workflow
from config.loader import config
import cv2

model_path = config['DEFAULT_MODEL_PATH']
test_image_path = os.path.join('input','example.jpg')

def case_01():
    test_image = cv2.imread(test_image_path)
    segmenter = segmentation.init_segmenter(model_path,'image')
    mask = segmentation.get_segmented_frame(test_image,segmenter)

    result = workflow.remove_background(test_image,mask)

    cv2.imshow('Final',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_workflow():
    input_path = os.path.join('input','video-input.mp4')
    output_path = os.path.join('output','debug')
    model_path = config['DEFAULT_MODEL_PATH']
    remover.main(input_path,output_path,model_path)