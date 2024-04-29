import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


def run_tests(face_recognizer, model_path='./model.keras'):
    test_cases = [
        ("./test/angelinajolie.jpg", "Angelina Jolie"),
        ("./test/sandra_bullock.jpg", "Sandra Bullock"),
        ("./test/meganfox.jpg", "Megan Fox"),
        ("./test/jenlawrence.jpg", "Jeniffer Lawrence"),
        ("./test/natalieportman.jpg", "Natalie Portman")
    ]

    for img_path, expected_celebrity in test_cases:
        test_img = preprocess_image(img_path, (224, 224))
        predictions = model.predict(test_img)
        prediction_index = np.argmax(predictions[0])
        predicted_name = face_recognizer.get_class_name(prediction_index)
        print(f"Expected: {expected_celebrity}, predicted: {predicted_name}")


def preprocess_image(image_path, image_size):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array
