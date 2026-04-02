from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)

# -----------------------------
# Download model from Google Drive
# -----------------------------
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    file_id = "1Icz6QF7OAWK8re8lNkmecVKcLUSbmlAq"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

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

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    try:
        img = Image.open(filepath).convert("RGB")
        img = img.resize((224,224))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)

        index = np.argmax(prediction)
        result = classes[index]

        confidence = float(np.max(prediction))*100
        confidence = round(confidence,2)

        # unknown detection
        top2 = np.sort(prediction[0])[-2:]
        difference = top2[1] - top2[0]

        if confidence < 70 or difference < 0.05:
            return render_template(
                "result.html",
                prediction="Invalid / Unsupported Image",
                image=filepath,
                damage=0,
                confidence=confidence,
                treatment="Upload clear crop leaf image only.",
                risk="Low"
            )

        if "healthy" in result.lower():
            damage = 0
        else:
            damage = round(confidence)

        if damage == 0:
            risk = "Low"
        elif damage < 40:
            risk = "Moderate"
        elif damage < 70:
            risk = "High"
        else:
            risk = "Very High"

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

    except:
        return render_template(
            "result.html",
            prediction="Error processing image",
            image=filepath,
            damage=0,
            confidence=0,
            treatment="Upload valid crop image.",
            risk="Low"
        )


if __name__ == "__main__":
    app.run(debug=True)
