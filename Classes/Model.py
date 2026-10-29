import os
import argparse
from keras import layers, models, utils, callbacks
import matplotlib.pyplot as plt

class LungModel:
    def __init__(self, data_dir: str = "Data"):
        self.class_names = ['COVID-19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
        self.data_dir = data_dir

        # Ensure runs directory exists
        os.makedirs("runs", exist_ok=True)

        # Datasets
        train_path = os.path.join(self.data_dir, "train")
        test_path = os.path.join(self.data_dir, "test")
        val_path = os.path.join(self.data_dir, "val")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data directory not found at: {train_path}")

        self.train_dataset = utils.image_dataset_from_directory(
            train_path,
            image_size=(224, 224),
            batch_size=64
        )
        self.test_dataset = utils.image_dataset_from_directory(
            test_path,
            image_size=(224, 224),
            batch_size=64
        ) if os.path.exists(test_path) else None

        self.validation_dataset = utils.image_dataset_from_directory(
            val_path,
            image_size=(224, 224),
            batch_size=64
        ) if os.path.exists(val_path) else None

        # Normalize Dataset
        self.normalization_layer = layers.Rescaling(1. / 255)

        self.train_dataset = self.train_dataset.map(lambda x, y: (self.normalization_layer(x), y))
        if self.validation_dataset is not None:
            self.validation_dataset = self.validation_dataset.map(lambda x, y: (self.normalization_layer(x), y))
        if self.test_dataset is not None:
            self.test_dataset = self.test_dataset.map(lambda x, y: (self.normalization_layer(x), y))

        # Create model
        self.model = models.Sequential([
            # Convolutional Blocks
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            # Fully Connected Head
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(4, activation='softmax')
        ])

        # Compile
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, epoch_size: int = 15):
        callback = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=epoch_size,
            callbacks=[callback]
        )

        self.visualize_train(history)
        self.save_model()

    def visualize_train(self, history):
        acc = history.history.get('accuracy', [])
        val_acc = history.history.get('val_accuracy', [])
        loss = history.history.get('loss', [])
        val_loss = history.history.get('val_loss', [])

        epochs_range = range(len(acc))

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Train Accuracy')
        if val_acc:
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        if val_loss:
            plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join("runs", "training_history.png"))
        plt.close()

    def save_model(self):
        save_path = os.path.join('runs', 'best_weight.h5')
        self.model.save(save_path)
