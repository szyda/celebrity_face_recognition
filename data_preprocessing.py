from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessing:
    def __init__(self, data_directory, image_size=(244, 244), batch_size=32):
        self.data_directory = data_directory
        self.image_size = image_size
        self.batch_size = batch_size

    """
    Generator danych z opcjonalną augmentacją dla danych treningowych. 
    Augumentacja to proces tworzenia zmodyfikowanego obrazu (np. poprzez odwrócenie, 
    czy odbicie lustrzane). Doda nam to różnorodności do zbioru danych. 
    """
    def initialize_datagen(self, augment=False):
        if augment:
            data = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
        else:
            data = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )

        return data

    """
    Funkcja wczytuje dane dla wybranego podzbioru: training/validation
    """
    def load_data(self, subset):
        datagen = self.initialize_datagen(augment=(subset == 'training')) # augumentujemy tylko zbiory treningowe (najbezpieczniej imo)

        return datagen.flow_from_directory(
            self.data_directory,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=subset
        )

    """
    Pomocnicze funkcje / helpers, które ułatwią dostęp do danych treningowych/walidacyjnych
    """

    def get_train_data(self):
        return self.load_data('training')

    def get_validation_data(self):
        return self.load_data('validation')
