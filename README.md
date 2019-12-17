# Accenter

## Build and Run Specifications
Use the `requirements.txt` file to properly set up Python virtual environment with `pip install -r requrements.txt`.

### Directory Setup
For functions in `dataUtil` to work as intended, the following directory setup is recommended.

    .
    ├── ...
    ├── data
    │   ├── processed
    │   │   ├── american
    │   │   │   └── clips
    │   │   ├── chinese
    │   │   └── ...
    │   └── raw
    │       ├── american
    │       ├── chinese
    │       └── ...
    └── ...
    
where each of `american`, `chinese`, `british`, etc. are considered "labels" for the classification
task and contain subdirectories `clip` within each of their processed folders.

### Dependencies
`pydub` throws warnings without an installation of `ffmpeg`, tools for manipulating multimedia files.
Visit the `pydub` [repository](https://github.com/jiaaro/pydub#getting-ffmpeg-set-up) to properly set up `ffmpeg`.

Use `pipreqs .` to generate `requirements.txt` based on all imports with versions used in import statements in the
project. 

## Classification
We implemented three models for classifying our three accent classes. All three can be found in `models/classification`. We trained the models on 15 epochs of our data and received results that can be seen in our final report.

## Preprocessing
We first split audio data into 1 second segments, cutting out the non-speaking time as well. To do this, run `python main.py segment <raw_data_dir> <output_data_dir>` where `raw_data_dir` is the directory your raw audio files are in, and `output_data_dir` is the directory the segmented pieces will be extracted to with a matching internal file structure.

For the final model, we run an MFCC preprocessing step to extract features instead of training on the raw audio wave files. To do this on a set of data setup as show above in directory setup, run `python main.py fextr <processed_dir>` where `processed_dir` is the directory with all of the segmented audio clips.

## Web Application
A simple Flask web app was built to demo Accenter with custom audio recordings. To start the Flask app, run
```
export FLASK_APP=app.py
flask run
```
to start the application locally.
