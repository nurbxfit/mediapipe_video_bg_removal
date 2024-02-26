import time 
from utils.remover import process_video
from utils import pipeline
from server import HttpServer

VIDEO_INPUT_PATH='input/video-input.mp4'
# VIDEO_INPUT_PATH='input/kid-dance.mp4'

def run():
    # start_time = time.time()
    # process_video(VIDEO_INPUT_PATH)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # formatted_time = "{:.2f} seconds".format(elapsed_time)
    # print("Program execution time:", formatted_time)
    httpServer = HttpServer.create_app()
    httpServer.run()
    # pipeline.main()


if __name__ == '__main__':
    run()