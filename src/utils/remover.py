from utils import video, segmentation, workflow
import concurrent.futures as cf
from tqdm import tqdm
import time
from config.loader import config

def run_remove_bg_process(input_path,output_path,model_path, input_type = 'video'):

    segmenter = segmentation.init_segmenter(model_path,'image')
    if input_type == 'video':
        run_video_processing(input_path,output_path,segmenter)
    else: 
        run_image_processing(input_path,output_path,segmenter)
    

def run_image_processing(input_path,output_path,segmenter):
    print('not yet implement')

def run_video_processing(input_path, output_path,segmenter):
    try:
        cap = load_local_video(input_path)
        width,height = video.get_video_info(input_path)
        processed_frames = []

        run_multi_threaded(cap,segmenter,processed_frames)

        processed_frames.sort(key=lambda x:x[1])
        video.write_frames(processed_frames,width,height,output_path)

        return output_path
    except Exception as e:
        print(f"Faile to process video:{e}")
        return None
    
def run_multi_threaded(frames, segmenter, processed_frames):
    with cf.ThreadPoolExecutor(max_workers=config['MAX_REMOVAL_WORKERS']) as executor:
        futures = []
        for frame, _ in frames:
            future = executor.submit(process_frame,frame,segmenter)
            futures.append(future)

        with tqdm(desc="Processing video frames", unit=" frames") as pbar:
            for future in cf.as_completed(futures):
                result = future.result()
                processed_frames.append(result)
                pbar.update(1)

def process_frame(frame, segmenter):
    try:
        timestamp = time.time()
        mask = segmentation.get_segmented_frame(frame,segmenter,timestamp)
        processed_frame = workflow.remove_background(frame,mask)

        return processed_frame, timestamp
    except Exception as e:
        print(f'Failed to remove background:{e}')
        raise e

def load_local_video(path):
    cap = video.read_video(path)
    return cap


# this is unecessary
def main(input_path,output_path,model_path):
    run_remove_bg_process(input_path,output_path,model_path)

if __name__ == '__main__':
    main()