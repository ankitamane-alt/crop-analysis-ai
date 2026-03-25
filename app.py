from flask import Flask, render_template, request
import os
from model import predict_image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if not os.path.exists("static"):
        os.makedirs("static")

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    result, confidence = predict_image(filepath)

    return render_template(
    "result.html",
    prediction=result,
    confidence=round(confidence,2),
    image='/' + filepath
 )
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
