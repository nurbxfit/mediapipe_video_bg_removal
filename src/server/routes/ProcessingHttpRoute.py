from flask_restful import Resource
from server.validation import validate_remove_bg_req
from flask import request
from config.loader import config
from threading import Thread
from utils.remover import run_remove_bg_process
from utils import fileUtils
from werkzeug.exceptions import NotFound
import os 

# Remove Background Controller
UPLOAD_FOLDER = config['DEFAULT_UPLOAD_FOLDER']
OUTPUT_FOLDER = config['DEFAULT_OUTPUT_FOLDER']
MODEL_PATH = config['DEFAULT_MODEL_PATH']

bg_thread = None

class RemoveBackground(Resource):
    def post(self):
        video_id = validate_remove_bg_req(request)

        print(f'Starting Background removal process for : {video_id}')
        source_file = os.path.join(UPLOAD_FOLDER,video_id,'input.mp4')

        if not fileUtils.file_exist(source_file):
            raise NotFound(f'Failed to process, file with id: {video_id} not found!')
        
        output_path = os.path.join(OUTPUT_FOLDER,video_id)

        # run some processing function (not blocking the return response)
        # Start processing in a separate thread
        bg_thread = Thread(target=run_remove_bg_process, kwargs={
            'input_path': source_file,
            'output_path': output_path,
            'model_path': 'models/selfie_segmenter.tflite',
        })
        bg_thread.start()
        return {'message': 'Video processing started',  'id':video_id}

    def cleanup(self):
        # Perform cleanup tasks after processing is complete
        # For example, deleting temporary files or logging
        bg_thread.join()
        pass
