from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH = 'C:/Users/rafis/OneDrive/Desktop/miniproject/mini1/Model/wristfinal.h5'
model = load_model(MODEL_PATH)

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "the bone is not fractured by wrist", 'fractured.html'
    
    else:
        preds = "The bone is  fractured by wrist", 'not_fractured.html'
    return preds

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')
     
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('C:/Users/rafis/OneDrive/Desktop/miniproject/mini1/static/uploads', filename)
            file.save(file_path)
            pred, output_page = model_predict(img_path=file_path)
            return render_template(output_page, pred_output=pred, user_image=file_path[50:])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5250, debug=True)
