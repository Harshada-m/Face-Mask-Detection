from flask import Flask, redirect, url_for, request, render_template, jsonify
import numpy as np
import threading
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
import os

model = tf.keras.models.load_model("C:/Users/dell/Azure Demo/Face Mask Detection/Mask_detection_model.h5")

os.environ["FLASK_ENV"] = "development"    # Creating A Devlopment Environment

app = Flask(__name__)       # Initilize App   

model = load_model('C:/Users/dell/Azure Demo/Face Mask Detection/Mask_detection_model.h5')

@app.route('/')
def index():
    # Main page
    return render_template('index1.html')  #Rendering Our Main Template

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        image = Image.open(request.files['file'].stream)     # Opening Uploaded File
        image = image.resize((224,224))                      # Resizing Image Based On Model Requirement
        image = img_to_array(image)                          # Changing Image to Array of pixels
        image = preprocess_input(image)                      # Preprocessing Image
        image=np.expand_dims(image, axis=0)                 

        
        predictions=model.predict(image)     # Predicting on Test Image
        predictions=predictions.reshape(-1)
        threshold=0.5
        y_pred=np.where(predictions >= threshold, 'Non Mask','Mask')
        result = y_pred[0]
        
        print('[PREDICTED CLASSES]: {}'.format(y_pred))
        print('[RESULT]: {}'.format(result))

        return result


if __name__ == '__main__':
  #  Start the Flask server in a new thread
  threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()
