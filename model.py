import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import InputLayer

# load once
model = tf.keras.models.load_model(
    "model.h5",
    custom_objects={"InputLayer": InputLayer},
    compile=False
)

classes = ["Apple___Apple_scab","Apple___Black_rot","Apple___healthy"]  # keep your full list

def predict_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)

    return classes[index], float(np.max(prediction))*100
