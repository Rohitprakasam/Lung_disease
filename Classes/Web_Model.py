from PIL import Image
import numpy as np
from keras import models

class Web_Model:
    def __init__(self, model_path, classes):
        self.model = models.load_model(model_path)
        self.classes = classes

    def preprocess_image(self, image):
        image = image.resize((224, 224))  # Resize to required input size
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def predict_with_probs(self, image):
        processed = self.preprocess_image(image)
        prediction = self.model.predict(processed, verbose=0)[0]
        max_idx = int(np.argmax(prediction))
        predicted_class = self.classes[max_idx]
        confidence = float(prediction[max_idx])
        probabilities_dict = {
            self.classes[i]: float(prediction[i])
            for i in range(len(self.classes))
        }
        return predicted_class, confidence, probabilities_dict

    def predict(self, image):
        predicted_class, _, _ = self.predict_with_probs(image)
        return predicted_class
