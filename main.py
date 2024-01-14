import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image


class CatDetector:
    def __init__(self):
        self.model = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/4")
        ])

        self.labels_path = "ImageNetLabels.txt"
        with open(self.labels_path) as f:
            self.labels = f.readlines()
        self.labels = [label.strip() for label in self.labels]

        # Define the cat breeds to detect
        self.cat_breeds = ['persian', 'tabby', 'siamese', 'ragdoll', 'sphynx']

    def preprocess_image(self, image):
        image = image.resize((224, 224))
        image = np.array(image) / 255.0  # Normalize the pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def predict(self, image_path):
        image = Image.open(image_path)
        preprocessed_image = self.preprocess_image(image)
        predictions = self.model.predict(preprocessed_image)
        predicted_labels = np.argsort(predictions[0])[-5:][::-1]  # Top 5 predicted labels
        print(predictions)
        for i in predicted_labels:
            print(self.labels[i])
        detected_cats = []
        accuracies = []
        for label in predicted_labels:
            label_text = self.labels[label].lower()
            for cat_breed in self.cat_breeds:
                if cat_breed in label_text:
                    detected_cats.append(self.labels[label].split(',')[0])
                    accuracies.append(predictions[0][label])
                    break

        return detected_cats, accuracies


# Usage
detector = CatDetector()


def detect_cats(file_name):
    detected_cats, accuracies = detector.predict(file_name)

    if detected_cats:
        print("Cats detected in the image:")
        for cat, accuracy in zip(detected_cats, accuracies):
            print(f"Cat: {cat}, Accuracy: {accuracy:.2f}")
    else:
        print("No cats found in the image.")


while True:
    print("----------------------------------------------------------------")
    file = input("Enter image file name:")
    if file == 'q':
        break
    detect_cats(file)
