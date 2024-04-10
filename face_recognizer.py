from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
import numpy as np
import scipy

class FaceRecognizer:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1):
        self.num_classes = num_classes
        self.model = self.build_model(input_shape, self.num_classes)

    def build_model(self, input_shape, num_classes):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False  # freeze the resnet50 base to not update weights

        model = Sequential([
            base_model,
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_data, val_data, epochs=10):
        checkpoint = keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=[checkpoint]
        )
        return history

    def predict(self, preprocessed_image):
        """Predicts the class of a preprocessed image.

        Args:
            preprocessed_image (numpy.ndarray): Preprocessed image ready for model input.

        Returns:
            int: The predicted class index, indicating the most likely class the image belongs to.
        """
        prediction = self.model.predict(preprocessed_image)
        return np.argmax(prediction, axis=1)

    def get_class_name(self, index):
        return {v: k for k, v in self.class_indices.items()}[index]