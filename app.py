from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)

# ✅ Download model if not exists
MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1Icz6QF7OAWK8re8lNkmecVKcLUSbmlAq"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ❗ DO NOT load model at start (prevents crash)
model = None

# Dataset classes
classes = [
"Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
"Blueberry___healthy","Cherry_(including_sour)___healthy","Cherry_(including_sour)___Powdery_mildew",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
"Corn_(maize)___healthy","Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot",
"Grape___Esca_(Black_Measles)","Grape___healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
"Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight",
"Potato___healthy","Potato___Late_blight","Raspberry___healthy","Soybean___healthy",
"Squash___Powdery_mildew","Strawberry___healthy","Strawberry___Leaf_scorch",
"Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___healthy",
"Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
"Tomato___Tomato_mosaic_virus","Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    global model

    # ✅ Load model only when needed (prevents crash)
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded!")

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    # Create static folder if not exists
    if not os.path.exists("static"):
        os.makedirs("static")

    # Save image
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Image preprocessing
    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    index = np.argmax(prediction)
    result = classes[index]

    confidence = float(np.max(prediction)) * 100
    confidence = round(confidence, 2)

    # Damage estimation
    if "healthy" in result.lower():
        damage = 0
    else:
        damage = round(confidence)

    # Risk level
    if damage == 0:
        risk = "Low"
    elif damage < 40:
        risk = "Moderate"
    elif damage < 70:
        risk = "High"
    else:
        risk = "Very High"

    # Treatment
    if "healthy" in result.lower():
        treatment = "No treatment required. Plant is healthy."
    elif "blight" in result.lower():
        treatment = "Apply fungicide spray and remove infected leaves."
    elif "rust" in result.lower():
        treatment = "Use sulfur-based fungicide."
    elif "mildew" in result.lower():
        treatment = "Improve air circulation and apply fungicide."
    else:
        treatment = "Consult agricultural expert."

    return render_template(
        "result.html",
        prediction=result,
        image=filepath,
        damage=damage,
        confidence=confidence,
        treatment=treatment,
        risk=risk
    )

# ✅ FIXED RUN (IMPORTANT FOR RENDER)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
