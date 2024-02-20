import time 
from utils.remover import remove_bg

VIDEO_INPUT_PATH='input/video-input.mp4'

def run():
    start_time = time.time()
    remove_bg(VIDEO_INPUT_PATH)
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = "{:.2f} seconds".format(elapsed_time)
    print("Program execution time:", formatted_time)