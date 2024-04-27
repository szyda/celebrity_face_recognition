from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time
import datetime


class FaceRecognizer:
    def __init__(self, input_shape=(224, 224, 3), num_classes=17, class_indices=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model(input_shape, self.num_classes)
        self.class_indices = class_indices

    @staticmethod
    def build_model(input_shape, num_classes):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

        model = Sequential([
            base_model,
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_data, val_data, epochs=10):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='500,520')

        checkpoint = keras.callbacks.ModelCheckpoint(
            'model_checkpoint.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        start_time = time.time()
        history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=[checkpoint, tensorboard_callback]
        )

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds.")

        self.model.save('model.h5')
        print("Model saved.")

        return history

    def predict(self, preprocessed_image):
        prediction = self.model.predict(preprocessed_image)
        predicted_index = np.argmax(prediction, axis=1)[0]
        print("Prediction output:", prediction)

        return predicted_index

    def get_class_name(self, index):
        class_name = self.class_indices.get(index, "Unknown celebrity")
        return class_name

    def print_classes(self):
        for c in self.class_indices:
            print(c)
