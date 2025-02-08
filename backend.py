#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from io import BytesIO
import warnings
from flask_cors import CORS


warnings.simplefilter(action='ignore', category=UserWarning)

app = Flask(__name__)

# Enabling CORS
CORS(app)


# Loading the color model
model = joblib.load("knn_a_color_model.pkl")

def image_to_rgb_array(image_data):
    # Openning the image
    img = Image.open(BytesIO(image_data)).convert("RGB")  # Ensure it's in RGB mode
    
    # Converting to numpy array
    rgb_array = np.array(img, dtype=np.uint8)  # Ensure correct data type
    
    return rgb_array.tolist()  

def get_dominant_colour(image_path):
    image_path = image_path
    rgb_array = image_to_rgb_array(image_path)
    #print(rgb_array)
    colours = []
    for i in rgb_array[0]:
        colours.append(model.predict([i])[0])
    #print(colours)
    
    return f"{max(set(colours), key=colours.count)}"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_data = image_file.read()
    dominant_color = get_dominant_colour(image_data)
    
    return jsonify({"dominant_color": dominant_color})

if __name__ == "__main__":
    app.run(debug=True)






