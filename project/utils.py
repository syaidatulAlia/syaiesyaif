from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = [label.strip() for label in open("labels.txt", "r").readlines()]

def process_image(image_path):
    # Create a numpy array of the right shape for the model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)

    # Normalize the image array
    normalized_image_array = (image_array / 127.5) - 1
    data[0] = normalized_image_array

    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
