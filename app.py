# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 23:07:41 2024

@author: PRATHAMESH MANDIYE
"""

from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
from waitress import serve

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['POST'])
def after():
    img = request.files['file1']
    img.save('static/file.jpg')

    # Face detection using Haar Cascade
    img1 = cv2.imread('static/file.jpg', 0)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = cascade.detectMultiScale(img1, 1.1, 3)

    cropped = None  # Initialize cropped variable

    for x, y, w, h in faces:
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = img1[y:y + h, x:x + w]

    if cropped is not None:
        cv2.imwrite('static/after.jpg', cropped)

        # Resize and preprocess image for the model
        image = cv2.imread('static/file.jpg', 0)
        image = cv2.resize(image, (48, 48))
        image = np.reshape(image, (1, 48, 48, 1))

        # Load the model and make predictions
        model = load_model('model2.h5')
        prediction = model.predict(image)
        label_map = ["Mouth Open", "Mouth Closed"]
        prediction = np.argmax(prediction)
        final_prediction = label_map[prediction]

        return render_template('after.html', data=final_prediction)
    else:
        return render_template('after.html', data="Face not detected")

if __name__ == "__main__":
    serve(app, host="0.0.0.0",port=8000)
