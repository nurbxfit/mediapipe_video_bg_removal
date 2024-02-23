import time 
from utils.remover import process_video
from utils.pipeline import main
VIDEO_INPUT_PATH='input/video-input.mp4'
# VIDEO_INPUT_PATH='input/kid-dance.mp4'

def run():
    # start_time = time.time()
    # process_video(VIDEO_INPUT_PATH)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # formatted_time = "{:.2f} seconds".format(elapsed_time)
    # print("Program execution time:", formatted_time)

    main()


if __name__ == '__main__':
    run()