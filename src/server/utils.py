import os
from config.loader import config
from utils import fileUtils
import hashlib



def get_video_path(video_id):
    video_folder = os.path.join(config['DEFAULT_OUTPUT_FOLDER'],str(video_id))
    video_path = os.path.join(video_folder,'output.mp4')
    return video_folder, video_path

def generate_video(video_path):
    with open(video_path, 'rb') as video_file:
        while True:
            data = video_file.read(4096)  # Adjust chunk size as needed
            if not data:
                break
            yield data

def get_ext(file_name):
    _,file_extension = os.path.splitext(file_name)
    return file_extension[1:] # remove "." in ".mp4"

def get_file_hash(file):
    return hashlib.sha256(file.read()).hexdigest()[:25]


def check_hash_file_existed(hash_value,output_folder, ext='mp4'):
    file_path = os.path.join(output_folder,hash_value,"input.mp4")
    return fileUtils.file_exist(file_path)

def save_video(video, save_file_path):
    # ensure save directory exist
    save_file_dir = os.path.dirname(save_file_path)
    # print(f"save_file_dir:{save_file_dir}")
    fileUtils.assert_dir(save_file_dir)
    # Seek to the beginning of the file
    video.seek(0)
    video.save(save_file_path)