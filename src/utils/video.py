import ffmpeg
import os
import numpy as np
import cv2
import time
from tqdm import tqdm

def get_video_info(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = get_video_stream(probe['streams'])
        width = video_stream['width']
        height = video_stream['height']
        return width, height
    except ffmpeg.Error as e:
        print("ffmpeg_error:", e.stderr)
        return
    
def get_video_stream(streams):
    for stream in streams:
        if stream['codec_type'] == 'video':
            return stream
    raise Exception("Failed to find video stream!")


def create_output_process(destination_path, width, height, frame_rate=24, audio_file_path=None):
    os.makedirs(destination_path, exist_ok=True)
    output_file_path = os.path.join(destination_path, 'output.mp4')

    ffmpeg_input = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgba', s='{}x{}'.format(width, height))

    if audio_file_path:
        ffmpeg_audio_input = ffmpeg.input(audio_file_path)
        output_process = (
            ffmpeg.concat(ffmpeg_input, ffmpeg_audio_input, v=1, a=1)
            .output(output_file_path, vcodec='libx264', pix_fmt='yuv420p', r=frame_rate, acodec='aac', strict='experimental',shortest=None)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    else:
        output_process = (
            ffmpeg_input
            .output(output_file_path, vcodec='libx264', pix_fmt='yuv420p', r=frame_rate, acodec='aac', strict='experimental')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    return output_process

def read_video(video_path,target_frame_rate=24):
    # Input video file with adjusted buffer size
    width, height = get_video_info(video_path)
    process = (
        ffmpeg.input(video_path, probesize=2*1024*1024)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=target_frame_rate)
        .run_async(pipe_stdout=True)
    )
    timestamp = int(time.time_ns() // 1_000_000)
    while True:
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        yield frame, timestamp
        timestamp = int(time.time_ns() // 1_000_000)
    process.communicate()
    

def read_video_cv2(video_path,target_frame_rate=24):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception('Failed to open video!')
    
    cap.set(cv2.CAP_PROP_FPS, target_frame_rate)
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            raise Exception('Failed to read video!')
        
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        yield frame, timestamp


def parse_frame_rate(frame_rate_str): # eg: "60/1" 
    numerator, denominator = map(int, frame_rate_str.split('/'))
    return numerator / denominator


def write_frames(frames,width,height,output_path):
    ffmpeg_process = create_output_process(output_path, width, height)
    
    with tqdm(desc="Saving video frames", unit=" frames") as pbar:
        for frame, _ in frames:
            if frame is None:
                pass
            else:
                ffmpeg_process.stdin.write(frame)
                pbar.update(1)

    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()