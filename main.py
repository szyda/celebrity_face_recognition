import os
from data_preprocessing import DataPreprocessing
from face_recognizer import FaceRecognizer


def main():
    # Inicjalizacja klas
    data_prep = DataPreprocessing()
    num_classes = data_prep.num_classes
    class_indices = data_prep.class_indices
    print("Number of classes found: {}\n".format(num_classes))
    print("Classes found:")

    face_recognizer = FaceRecognizer(num_classes=num_classes, class_indices=class_indices)

    # Przygotowanie zdjec do trenowania - ominac jesli juz istnieja
    if not os.path.isdir("./data/processed"):
        print("Preprocessing images and detecting faces...")
        data_prep.crop_faces()
    else:
        print("Skipping preprocessing images")

    # Wczytanie danych
    print("Loading training and validation data...")
    train_data = data_prep.get_train_data()
    val_data = data_prep.get_validation_data()

    # Trenowanie modelu
    print("Training the model...")
    history = face_recognizer.train(train_data=train_data, val_data=val_data, epochs=20)

    # Podsumowanie pracy modelu
    print("Training completed. Model performance:")
    print(f"Accuracy: {history.history['accuracy'][-1]},\nLoss: {history.history['loss'][-1]}")

    # Test
    test_image_path = './test.jpg'
    test_img_preprocessed = data_prep.preprocess_image(test_image_path)
    prediction_index = FaceRecognizer.predict(test_img_preprocessed)
    print("Prediction index: ",prediction_index)
    predicted_class_name = FaceRecognizer.get_class_name(prediction_index)
    print(f"Predicted class name: {predicted_class_name}")


    # TO DO:
    # Dodac testy


if __name__ == '__main__':
    main()
