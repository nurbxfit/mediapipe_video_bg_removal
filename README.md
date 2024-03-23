# About

Simple script demo, on how to remove background from a video.
It use [google mediapipe segmentation solution](https://developers.google.com/mediapipe/solutions/vision/image_segmenter#configurations_options), to perform selfie segmentation on the video
then it seperate the background from the segmented foreground.

# Requirements

1. [FFmpeg](https://www.ffmpeg.org/download.html)
2. opencv

# How to run

1. cd into this directory

```
cd py_video_bg_remover
```

2. create new virtual environment

```
python3 -m venv .venv
```

3. get into the environment

```
 .\.venv\Scripts\activate
```

4. install dependencies

```
pip install -r .\requirements.txt
```

5. run the script

```
cd src && python main.py
```

6. get out of venv

```
deactivate
```

# Updates

## (23/2/2024) Threadpool

- I am able to improve the processing time by using threadpool
- only takes 30 seconds when using selfie_segmenter model
- only takes 2.3 minutes when using selfie_multiclass_256x256 model
- able to remove jitter/glitch on output video when using threadpools.

## (26/2/2024) Http Server

- added simple HttpServer to upload video and query processed video

```
cd src && python main.py
```

then can open postman and upload video to `http://localhost:5000/upload`
then can start processing by provide the video_id to `http://localhost:5000/remove-bg/video`
then can query back the processed video at `http://localhost:5000/video/<videoId>`

- sever can be run using docker (currently not working)

```
docker build -t video-server .
```

```
docker run -p 5000:5000 --name video-server video-server
```

# Example 

### Input 
![input image](/screenshots/input.png)

### Output
![output image](/screenshots/output.png)

The output is not perfect because it depends on the mediapipe segmentation model. but you can tune the result using opencv by looking into the file `src/utils/workflow.py`.