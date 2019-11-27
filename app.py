from flask import Flask, render_template, request
import subprocess
import os

template_dir = os.path.abspath('./web/templates')
static_dir = os.path.abspath('./web/static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)


@app.route('/')
def init_accenter():
    return render_template('Accenter.html')


