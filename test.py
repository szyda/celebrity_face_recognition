import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def run_tests(face_recognizer, model_path='best_model.keras', test_data_directory='./test/'):
    test_cases = [
        ("./test/angelina_jolie.jpg", "Angelina Jolie"),
        ("./test/sandra_bullock.jpg", "Sandra Bullock"),
        ("./test/meganfox.jpg", "Megan Fox"),
        ("./test/jenlawrence.jpg", "Jeniffer Lawrence"),
        ("./test/natalieportman.jpg", "Natalie Portman")
    ]

    for img_path, expected_celebrity in test_cases:
        test_img = preprocess_image(img_path, (224, 224))
        prediction_index = face_recognizer.predict(test_img)
        predicted_name = face_recognizer.get_class_name(prediction_index)
        assert predicted_name == expected_celebrity, f"Test failed for {img_path}. Expected {expected_celebrity}, got {predicted_name}"

    print("All tests passed.")

def preprocess_image(image_path, image_size):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    return img_array
