from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import tensorflow as tf
import json

app=Flask(__name__)
model = tf.keras.models.load_model('healthy_vs_rotten.h5')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/inspect')
def inspect():
    return render_template("inspect.html")

@app.route('/predict', methods=['GET', 'POST'])
def output():
    prediction = None
    if request.method == 'POST':
        if 'pc_image' not in request.files:
            return "No file uploaded"
        f=request.files['pc_image']
        img_path = "static/uploads/"+f.filename
        f.save(img_path)
        #Resize the image to the required size and convert it to RGB
        img=load_img(img_path, target_size=(224,224), color_mode='rgb')
        #Convert the image to an array and normalize it
        image_array = np.array(img)
        # Add a batch dimension
        image_array = image_array / 255.0 
        # Normalize pixel values to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)
        with open("class_indices.json", "r") as f:
            class_indices = json.load(f)
        index_to_class = {v: k for k, v in class_indices.items()}
        #Use the pre-trained model to make a prediction
        prediction_probs = model.predict(image_array)[0]
        pred_index = np.argmax(prediction_probs)
        confidence = float(np.max(prediction_probs))
        prediction = index_to_class[pred_index]
        print("Prediction:", prediction)
        print("Confidence:", confidence)

    return render_template("output.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=2222)