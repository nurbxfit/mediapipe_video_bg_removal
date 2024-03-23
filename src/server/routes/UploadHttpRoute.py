from flask_restful import Resource
from flask import request
from server.utils import get_file_hash, check_hash_file_existed, save_video
from server.validation import validate_upload
from config.loader import config
import os 
from werkzeug.exceptions import InternalServerError

UPLOAD_FOLDER=config['DEFAULT_UPLOAD_FOLDER']

class Upload(Resource):
    
    def post(self):
       # validate upload request        
        try:
            video, ext = validate_upload(request)

            # hash the video
            video_hash = get_file_hash(video)
            # check if the hash folder existed or not

            # if existed we do not upload, we tell them file uploaded already
            if check_hash_file_existed(video_hash,ext):
                return {'message': 'File uploaded successfully', 'uploaded': video.filename, 'video_id': video_hash}
            
            # if not exist we continue upload
            # fileUtils.cls_dir(UPLOAD_FOLDER)
            new_file_name = f'input.mp4'
            temp_save_file_path = os.path.join(UPLOAD_FOLDER,video_hash,new_file_name)
            save_video(video,temp_save_file_path)

            return {'message': 'File uploaded successfully', 'uploaded': video.filename, 'video_id': video_hash}

        except Exception as e:
            raise InternalServerError(str(e))

    
    def cleanup(self):
        # Perform cleanup tasks after processing is complete
        # For example, deleting temporary files or logging

        pass
