from server.validation import validate_video_id
from server.utils import get_video_path, get_ext ,get_file_hash ,check_hash_file_existed, save_video
from werkzeug.exceptions import BadRequest
import os
from config.loader import config
from utils.remover import run_remove_bg_process


UPLOAD_FOLDER = config['DEFAULT_UPLOAD_FOLDER']
OUTPUT_FOLDER = config['DEFAULT_OUTPUT_FOLDER']
MODEL_PATH = config['DEFAULT_MODEL_PATH']

from fastapi import APIRouter, status, HTTPException, BackgroundTasks,UploadFile
from fastapi.responses import FileResponse

router = APIRouter(
    prefix='/api/v1/videos',
    tags=['videos']
)


@router.get('/{video_id}', status_code=status.HTTP_200_OK)
async def query_video(video_id: str):
    try:
        validate_video_id(video_id) 
        _, video_path = get_video_path(video_id) 
        if not os.path.exists(video_path):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Item not found matching the given ID')
    except BadRequest as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid video ID')
    
    return FileResponse(video_path)


@router.post('/{video_id}/remove-bg', status_code=status.HTTP_202_ACCEPTED)
async def remove_bg(video_id:str,background_tasks: BackgroundTasks):
    if not video_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='please provide valid video_id')
    
    video_dir = os.path.join(UPLOAD_FOLDER,video_id)
    if not os.path.exists(video_dir):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'video_id: {video_id} not found!')
    input_path = os.path.join(video_dir,'input.mp4')
    if not os.path.isfile(input_path):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Something went wrong, unable to read source: {video_id}')

    output_path = os.path.join(OUTPUT_FOLDER,video_id)
    background_tasks.add_task(run_remove_bg_process,input_path,output_path,MODEL_PATH)

    return {'message': 'Video processing started',  'id':video_id}

    
@router.post('/')
async def upload_video(video:UploadFile):
    if video is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='no video file provided')
    if video.filename == '':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='No video selected')
    
    file_ext = get_ext(video.filename)
    if not file_ext == 'mp4' and not file_ext == 'webm':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Only mp4 or webm file are supported")
    
    # get file hash
    video_hash = get_file_hash(video.file)
    # check if the hash folder existed or not
    # if existed we do not upload, we tell them file uploaded already
    if check_hash_file_existed(video_hash,file_ext):
        return {'message': 'File uploaded successfully', 'uploaded': video.filename, 'video_id': video_hash}
            
    # if not exist we continue upload
    new_file_name = f'input.mp4'
    temp_save_file_path = os.path.join(UPLOAD_FOLDER,video_hash,new_file_name)
    save_video(video,temp_save_file_path)
    
    return {'message': 'File uploaded successfully', 'uploaded': video.filename, 'video_id': video_hash}

