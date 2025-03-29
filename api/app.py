# from flask import Flask, request, jsonify
# from flask_cors import CORS  # Import CORS
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import os

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load the trained model
# model = tf.keras.models.load_model("../model/model.keras")

# # Class names corresponding to the labels
# class_names = ["vascular lesion", "squamous cell carcinoma", "seborrheic keratosis", 
#                "pigmented benign keratosis", "nevus", "melanoma", 
#                "dermatofibroma", "basal cell carcinoma", "actinic keratosis"]

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files["file"]
#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     try:
#         img = load_img(filepath, target_size=(180, 180))
#         img_array = img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         pred = model.predict(img_array)
#         predicted_class = class_names[np.argmax(pred)]

#         return jsonify({"prediction": predicted_class})
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model("../model/model.keras")

# Class names corresponding to the labels
class_names = ["vascular lesion", "squamous cell carcinoma", "seborrheic keratosis", 
               "pigmented benign keratosis", "nevus", "melanoma", 
               "dermatofibroma", "basal cell carcinoma", "actinic keratosis"]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img = load_img(filepath, target_size=(180, 180))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        predicted_class = class_names[np.argmax(pred)]

        return jsonify({"prediction": predicted_class})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
