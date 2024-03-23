from flask_restful import Resource
from server.validation import validate_video_id
from server.utils import get_video_path, generate_video
from werkzeug.exceptions import NotFound, BadRequest
from flask import make_response, stream_with_context
import os

class QueryVideo(Resource):
    def get(self,video_id):
        # validate video_id
        validate_video_id(video_id)
        _,video_path = get_video_path(video_id)

        print(f'video_path:{video_path}')
        try:
            # read file and add in the response
            with open(video_path,'rb'):
                response = make_response(
                    stream_with_context(generate_video(video_path)), 200
                )
                response.headers['Content-Type'] = 'video/mp4'
                response.headers['Content-Length'] = os.path.getsize(video_path)
                return response      

        except FileNotFoundError as e:
            print(f'FileNotFoundError:{e}')
            raise NotFound(f"Video with ID {video_id} not found")
            