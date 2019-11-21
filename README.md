# Accenter

## Build and Run Specifications
Use the `requirements.txt` file to properly set up Python virtual environment with `pip install -r requrements.txt`.

### Directory Setup
For functions in `dataUtil` to work as intended, the following directory setup is recommended.

    .
    ├── ...
    ├── data
    │   ├── processed
    │   │   ├── english
    │   │   │   └── clips
    │   │   ├── chinese
    │   │   └── ...
    │   └── raw
    │       ├── english
    │       ├── chinese
    │       └── ...
    └── ...
    
where each of `english`, `chinese`, `british`, etc. are considered "labels" for the classification
task and contain subdirectories `clip` within each of their processed folders.

### Dependencies
`pydub` throws warnings without an installation of `ffmpeg`, tools for manipulating multimedia files.

## Classification

## Conversion

## Preprocessing
