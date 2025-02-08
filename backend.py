#!/usr/bin/env python
# coding: utf-8

# In[4]:


from PIL import Image
import numpy as np
import joblib
from flask import Flask, request, jsonify
from io import BytesIO
import warnings
from flask_cors import CORS


warnings.simplefilter(action='ignore', category=UserWarning)

app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)


# Loading the color model
model = joblib.load("knn_a_color_model.pkl")

def image_to_rgb_array(image_data):
    # Open the image from the byte data
    img = Image.open(BytesIO(image_data)).convert("RGB")  # Ensure it's in RGB mode
    
    # Convert the image to a NumPy array
    rgb_array = np.array(img, dtype=np.uint8)  # Ensure correct data type
    
    return rgb_array.tolist()  # Convert to a list if needed

def get_dominant_colour(image_path):
    image_path = image_path
    rgb_array = image_to_rgb_array(image_path)
    #print(rgb_array)
    colours = []
    for i in rgb_array[0]:
        colours.append(model.predict([i])[0])
    #print(colours)
    
    return f"{max(set(colours), key=colours.count)}"

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


# In[ ]:




