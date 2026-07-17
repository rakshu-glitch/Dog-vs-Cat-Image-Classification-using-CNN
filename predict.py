import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os

model = tf.keras.models.load_model("model.keras")

class_names = ["Cat", "Dog"]

img_path = input("Enter image path: ").strip()

if not os.path.exists(img_path):
    print("Image not found!")
    exit()

img = image.load_img(img_path, target_size=(224,224))

img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array, verbose=0)

print("\nRaw Prediction:", prediction)

predicted = np.argmax(prediction)

print("\nPrediction:", class_names[predicted])
print(f"Confidence: {prediction[0][predicted]*100:.2f}%")