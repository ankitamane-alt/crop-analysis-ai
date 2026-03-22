from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Dataset classes
classes = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Blueberry___healthy",
"Cherry_(including_sour)___healthy",
"Cherry_(including_sour)___Powdery_mildew",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
"Corn_(maize)___Common_rust_",
"Corn_(maize)___healthy",
"Corn_(maize)___Northern_Leaf_Blight",
"Grape___Black_rot",
"Grape___Esca_(Black_Measles)",
"Grape___healthy",
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Orange___Haunglongbing_(Citrus_greening)",
"Peach___Bacterial_spot",
"Peach___healthy",
"Pepper,_bell___Bacterial_spot",
"Pepper,_bell___healthy",
"Potato___Early_blight",
"Potato___healthy",
"Potato___Late_blight",
"Raspberry___healthy",
"Soybean___healthy",
"Squash___Powdery_mildew",
"Strawberry___healthy",
"Strawberry___Leaf_scorch",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___healthy",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_mosaic_virus",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    # Save uploaded image
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Image preprocessing
    img = Image.open(filepath)
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # Model prediction
    prediction = model.predict(img)

    index = np.argmax(prediction)

    result = classes[index]

    confidence = float(np.max(prediction)) * 100
    confidence = round(confidence,2)

    # Damage estimation
    if "healthy" in result.lower():
        damage = 0
    else:
        damage = round(confidence)

    # Insurance risk level
    if damage == 0:
        risk = "Low"
    elif damage < 40:
        risk = "Moderate"
    elif damage < 70:
        risk = "High"
    else:
        risk = "Very High"

    # Treatment suggestions
    if "healthy" in result.lower():
        treatment = "No treatment required. Plant is healthy."
    elif "blight" in result.lower():
        treatment = "Apply fungicide spray and remove infected leaves."
    elif "rust" in result.lower():
        treatment = "Use sulfur-based fungicide."
    elif "mildew" in result.lower():
        treatment = "Improve air circulation and apply fungicide."
    else:
        treatment = "Consult agricultural expert for treatment."

    return render_template(
        "result.html",
        prediction=result,
        image=filepath,
        damage=damage,
        confidence=confidence,
        treatment=treatment,
        risk=risk
    )

if __name__ == "__main__":
    if __name__ == "__main__":
    app.run()