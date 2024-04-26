from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import face_recognition
import os
import cv2
import numpy as np
from datetime import datetime


class DataPreprocessing:
    def __init__(self, data_directory="./data/", image_size=(224, 224), batch_size=32):
        self.data_directory = data_directory
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes, self.class_indices = self.get_class_data()

    @staticmethod
    def initialize_datagen(augment=True):
        if augment:
            data_gen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.3,
                height_shift_range=0.3,
                shear_range=0.3,
                zoom_range=0.3,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.3  # 70% training 30% validation
            )
        else:
            data_gen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.3
            )

        return data_gen

    def load_data(self, subset):
        data_path = self.data_directory
        datagen = self.initialize_datagen(augment=(subset == 'training'))

        return datagen.flow_from_directory(
            data_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=subset
        )

    def get_class_data(self):
        classes = [d for d in os.listdir(self.data_directory) if os.path.isdir(os.path.join(self.data_directory, d)) and d != "processed"]
        classes.sort()
        class_indices = {idx: cls for idx, cls in enumerate(classes)}

        return len(classes), class_indices

    def get_train_data(self):
        return self.load_data('training')

    def get_validation_data(self):
        return self.load_data('validation')

    def crop_faces(self, directory="./data", overwrite=False):
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(subdir, file)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image. Skipping: {image_path}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(image)
                if faces:
                    top, right, bottom, left = faces[0]
                    face_image = image[top:bottom, left:right]
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    if overwrite:
                        cv2.imwrite(image_path, face_image)
                        print(f"Overwritten image at {image_path} with cropped face.")
                    else:
                        new_filename = f"{os.path.splitext(image_path)[0]}_face{os.path.splitext(image_path)[1]}"
                        cv2.imwrite(new_filename, face_image)
                        print(f"Saved cropped face image as {new_filename}")
                else:
                    print(f"No faces found in {image_path}. Skipping cropping.")

    @staticmethod
    def detect_faces(image):
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            return face_locations[0]
        return None
