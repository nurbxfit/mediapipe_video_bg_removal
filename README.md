# About

Simple script demo, on how to remove background from a video.
It use [google mediapipe segmentation solution](https://developers.google.com/mediapipe/solutions/vision/image_segmenter#configurations_options), to perform selfie segmentation on the video
then it seperate the background from the segmented foreground.

# How to run

1. cd into this directory

```
cd bgRemove
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
