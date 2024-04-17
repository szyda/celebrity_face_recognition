import os
from data_preprocessing import DataPreprocessing
from face_recognizer import FaceRecognizer
from test import run_tests


def main():
    data_prep = DataPreprocessing()
    num_classes = data_prep.num_classes
    class_indices = data_prep.class_indices

    print("Number of classes found: {}\n".format(num_classes))
    print("Classes found:")

    for index in class_indices:
        print(class_indices[index])

    face_recognizer = FaceRecognizer(num_classes=num_classes, class_indices=class_indices)

    if not os.path.isdir("data"):
        print("Preprocessing images and detecting faces...")
        data_prep.crop_faces()
    else:
        print("Skipping preprocessing images")

    print("Loading training and validation data...")
    train_data = data_prep.get_train_data()
    val_data = data_prep.get_validation_data()

    print("Training the model...")
    history = face_recognizer.train(train_data=train_data, val_data=val_data, epochs=30)

    print("Training completed. Model performance:")
    print(f"Accuracy: {history.history['accuracy'][-1]},\nLoss: {history.history['loss'][-1]}")

    print("\nRunning tests...")
    run_tests(face_recognizer)


if __name__ == '__main__':
    main()
