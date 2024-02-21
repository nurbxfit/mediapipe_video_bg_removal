import ffmpeg
import os
import numpy as np

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


def create_output_process(destination_path,width, height, frame_rate = 24):
    os.makedirs(destination_path,exist_ok=True)
    output_file_path = os.path.join(destination_path,'output.mp4')
    output_process = (
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgba', s='{}x{}'.format(width, height))
            .output(output_file_path, vcodec='libx264', pix_fmt='yuv420p', r=frame_rate,colorspace='bt709')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    return output_process

def read_video(video_path, width, height, target_frame_rate=24):
    # Input video file with adjusted buffer size
    process = (
        ffmpeg.input(video_path, probesize=2*1024*1024)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=target_frame_rate)
        .run_async(pipe_stdout=True)
    )

    while True:
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        yield frame

    process.communicate()


def parse_frame_rate(frame_rate_str): # eg: "60/1" 
    numerator, denominator = map(int, frame_rate_str.split('/'))
    return numerator / denominator


