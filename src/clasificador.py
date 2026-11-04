
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Clase creada para reutilizar el modelo

class MosquitoClassifier:

    def __init__(self, model_path):

        self.model = load_model(model_path)

        self.class_names = [
            "aegypti",
            "albopictus",
            "anopheles",
            "culex"
        ]

    def preprocess_image(self, image_path):

        img = image.load_img(
            image_path,
            target_size=(224, 224)
        )

        img_array = image.img_to_array(img)

        img_array = img_array.astype("float32")

        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        return img_array

    def predict(self, image_path):

        img_array = self.preprocess_image(
            image_path
        )

        predictions = self.model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(
            predictions
        )

        confidence = float(
            predictions[0][predicted_index]
        )

        predicted_class = self.class_names[
            predicted_index
        ]

        return {
            "class": predicted_class,
            "confidence": round(
                confidence,
                4
            )
        }