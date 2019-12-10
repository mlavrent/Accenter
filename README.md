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

## Conversion

## Preprocessing

## Web Application
A simple Flask web app was built to demo Accenter with custom audio recordings. To start the Flask app, run
```
export FLASK_APP=app.py
flask run
```
to start the application locally.
