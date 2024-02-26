from flask_restful import Resource, Api
from flask import request, make_response, Response, stream_with_context
from threading import Thread
from utils import fileUtils
from utils.pipeline import run_pipeline_service
import os 
from werkzeug.exceptions import NotFound, BadRequest
import re

UPLOAD_FOLDER= os.path.join('uploads')
EXTRACTED_FRAMES_FOLDER = 'output/extracted'
BACKGROUND_REMOVED_FOLDER = 'output/cleaned'
FINAL_VIDEO_OUTPUT = os.path.join('output/final')
FRAME_RATE = 24

class Upload(Resource):
    
    def post(self):
        # return {request}
        if 'file' not in request.files:
            return {'message': 'No file provided'}, 400
        
        file = request.files['file']

        if file.filename == '':
            return {'message': 'No selected file'}, 400
        
        file_ext = fileUtils.get_ext(file.filename)
        if not file_ext == 'mp4':
            return {'message': 'Only mp4 file are supported'}, 400 
        
        fileUtils.cls_dir(UPLOAD_FOLDER)
        file_id = fileUtils.gen_cuid()
        new_file_name = f'{file_id}.{file_ext}'
        SOURCE_FILE = f'{UPLOAD_FOLDER}/{new_file_name}'
        OUTPUT_FOLDER = f'output/{file_id}'
        file.save(SOURCE_FILE)
        # run some processing function (not blocking the return response)
        # Start processing in a separate thread
        bg_remove_thread = Thread(target=run_pipeline_service, kwargs={
            'FRAME_RATE': FRAME_RATE,
            'SOURCE_FILE': SOURCE_FILE,
            'OUTPUT_FOLDER': OUTPUT_FOLDER,
            'MODEL_PATH': 'models/selfie_segmenter.tflite',
        })
        bg_remove_thread.start()
        return {'message': 'File uploaded successfully', 'uploaded': file.filename, 'id':file_id}
    
    def cleanup(self):
        # Perform cleanup tasks after processing is complete
        # For example, deleting temporary files or logging
        pass


valid_chars = r"[0-9a-zA-Z]"
class Video(Resource):
    def get(self,video_id):
        # return {'Hello': 'world', 'id':video_id}
        
        # perform some input validation
        if len(video_id) != 25:
            raise BadRequest("Invalid video ID length")
        if not re.match(valid_chars, video_id):
            raise BadRequest("Invalid characters in video ID")
        
        # join video path using os.path instead of directly join as format string
        video_folder = os.path.join("output",str(video_id))
        video_path = os.path.join(video_folder,"output.mp4")
        try:
            # return {'Hello': 'world', 'id':video_id}
            with open(video_path, 'rb') as video_file:
                # Set appropriate headers
                response = make_response(
                    stream_with_context(generate_video(video_path)), 200
                )
                response.headers['Content-Type'] = 'video/mp4'
                response.headers['Content-Length'] = os.path.getsize(video_path)  # Add Content-Length header
                return response
        except FileNotFoundError:
            raise NotFound(f"Video with ID {video_id} not found")

def generate_video(video_path):
    with open(video_path, 'rb') as video_file:
        while True:
            data = video_file.read(4096)  # Adjust chunk size as needed
            if not data:
                break
            yield data