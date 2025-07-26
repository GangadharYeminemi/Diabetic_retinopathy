# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

# Dataset location
data_dir = r"D:\Projects\Diabetic_retinopathy\Diabetic_Retinopathy"
dr_dir = os.path.join(data_dir, "DR")
no_dr_dir = os.path.join(data_dir, "no_dr")

# Verify directories exist
print("DR directory exists:", os.path.exists(dr_dir))
print("No DR directory exists:", os.path.exists(no_dr_dir))

# Create DataFrame for binary classification
df = pd.DataFrame({
    "image_path": [os.path.join(dr_dir, f) for f in os.listdir(dr_dir)] +
                  [os.path.join(no_dr_dir, f) for f in os.listdir(no_dr_dir)],
    "label": ["DR"] * len(os.listdir(dr_dir)) + ["No_DR"] * len(os.listdir(no_dr_dir))
})

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_dataframe(
    df,
    x_col="image_path",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

validation_generator = train_datagen.flow_from_dataframe(
    df,
    x_col="image_path",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the model
model.save("binary_dr_cnn.keras")
print("Training completed. Model saved as 'binary_dr_cnn.keras'.")