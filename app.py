from flask import Flask, render_template, request
from matplotlib import pyplot as plt
import subprocess
import scipy.io.wavfile
import numpy as np
from dataUtil.featureExtraction import segment_and_extract
import os

template_dir = os.path.abspath('./web/templates')
static_dir = os.path.abspath('./web/static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)


@app.route('/')
def init_accenter():
    return render_template('Accenter.html')


@app.route("/classify", methods=['POST'])
def classify_accent():
    rawAudio = request.get_data()

    audioFile = open('recordedAudio.wav', 'wb')
    audioFile.write(rawAudio)
    audioFile.close()

    spectrogram, mfccs = segment_and_extract('recordedAudio.wav')
    if spectrogram is None:
        return "ERROR"
    else:
        accent = np.random.choice(["british", "english", "chinese", "korean"], 1)[0]
        print(accent)
        return accent.capitalize()
