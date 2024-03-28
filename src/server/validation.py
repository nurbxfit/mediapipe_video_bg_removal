import os
import re
from server.utils import get_ext

# Utilities functions
valid_chars = r"[0-9a-zA-Z]"
def validate_video_id(video_id):
    if len(video_id) != 25:
        raise Exception("Invalid video ID length")
    if not re.match(valid_chars, video_id):
        raise Exception("Invalid characters in video ID")
    return True

def validate_upload(request):
    if 'file' not in request.files:
        raise Exception('No file provided')
    video = request.files['file']
    if video.filename == '':
        raise Exception(f"No video selected")
    file_ext = get_ext(video.filename)
    if not file_ext == 'mp4' and not file_ext == 'webm':
        raise Exception(f"Only mp4 or webm file are supported")
    
    return video, file_ext

def validate_remove_bg_req(request):
    json_data = request.json
    if 'video_id' not in json_data:
        raise Exception('No video id provided!')
    
    return json_data['video_id']