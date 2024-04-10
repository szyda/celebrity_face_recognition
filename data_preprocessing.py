from tensorflow.keras.preprocessing.image import ImageDataGenerator
import face_recognition
import os, cv2
from datetime import datetime

class DataPreprocessing:
    def __init__(self, data_directory="./data/", image_size=(224, 224), batch_size=32):
        self.data_directory = data_directory
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes, self.class_indices = self.get_class_data()

    def initialize_datagen(self, augment=False):
        if augment:
            data_gen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2  # 80% training 20% validation
            )
        else:
            data_gen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )

        return data_gen

    def load_data(self, subset):
        datagen = self.initialize_datagen(augment=(subset == 'training'))
        return datagen.flow_from_directory(
            self.data_directory,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=subset
        )

    def get_class_data(self):
        classes = [d for d in os.listdir(self.data_directory) if os.path.isdir(os.path.join(self.data_directory, d))]
        classes.sort()
        class_indices = {cls: idx for idx, cls in enumerate(classes)}
        return len(classes), class_indices

    def get_train_data(self):
        return self.load_data('training')

    def get_validation_data(self):
        return self.load_data('validation')

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=self.image_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # reshaping to (1, height, width, channels)
        img_array /= 255.0  # normalization
        return img_array

    def crop_faces(self):
        for subdir, dirs, files in os.walk(self.data_directory):
            output_folder = os.path.join(subdir, "processed")
            os.makedirs(output_folder, exist_ok=True)

            for file in files:
                image_path = os.path.join(subdir, file)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image. Skipping: {image_path}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to rgb for face_recognition lib // not sure if needed?
                face = self.detect_faces(image)
                if face:
                    top, right, bottom, left = face
                    face_image = image[top:bottom, left:right]
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)  # convert to bgr for opencv lib
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    face_file_path = os.path.join(output_folder, f"face_{timestamp}.jpg")
                    cv2.imwrite(face_file_path, face_image)

    def detect_faces(self, image):
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            return face_locations[0]
        return None