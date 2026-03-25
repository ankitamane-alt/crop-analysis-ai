from flask import Flask, render_template, request
import os
from model import predict_image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']

        if not os.path.exists("static"):
            os.makedirs("static")

        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        prediction, confidence = predict_image(filepath)
        image_path = '/' + filepath

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence, 2) if confidence else None,
        image=image_path
    )


