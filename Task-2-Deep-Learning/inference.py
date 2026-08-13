import sys
import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

model = tf.keras.models.load_model("models/cifar10_cnn.keras")

# Example: replace this with an actual CIFAR-10-sized image array.
def predict(image):
    image = np.asarray(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    probabilities = model.predict(image, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return CLASS_NAMES[index], float(probabilities[index])

print("Model loaded. Use predict(image) with a 32x32x3 image array.")
