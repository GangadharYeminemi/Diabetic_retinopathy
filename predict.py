# predict.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model("binary_dr_cnn.keras")

# Function to preprocess and predict
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"File {img_path} does not exist.")
        return

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Output result
    if prediction >= 0.5:
        print(f"Prediction: No_DR (Confidence: {prediction:.2f})")
    else:
        print(f"Prediction: DR (Confidence: {1 - prediction:.2f})")

# Main block with hardcoded path
if __name__ == "__main__":
    image_path = r"D:\Projects\Diabetic_retinopathy\Diabetic_Retinopathy\no_dr\19-EYE-4245-Non-Proliferative-Diabetic-Retinopathy-CQD_jpg - Copy (4).webp"
    predict_image(image_path)
