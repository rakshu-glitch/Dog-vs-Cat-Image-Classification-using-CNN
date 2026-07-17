import os
import tensorflow as tf

dataset_path = "PetImages"
deleted = 0

for category in ["Cat", "Dog"]:
    folder = os.path.join(dataset_path, category)

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        try:
            # TensorFlow tries to decode the image
            image = tf.io.read_file(file_path)
            tf.image.decode_jpeg(image)

        except Exception:
            print("Deleting:", file_path)
            os.remove(file_path)
            deleted += 1

print(f"\nFinished! Deleted {deleted} corrupted images.")