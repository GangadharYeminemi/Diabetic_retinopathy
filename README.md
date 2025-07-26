 Diabetic Retinopathy Detection Using CNN

 Overview
This project uses a Convolutional Neural Network (CNN) to classify fundus images into **Diabetic Retinopathy** or **No_DR**.
 Methodology

- Image augmentation: rotation, zoom, shift, flip
- CNN Architecture: Conv2D → MaxPool → Dense → Dropout
- Training: Using Keras and validation split
- Model saved in Keras format

 Tech Stack
- TensorFlow, Keras, Pandas
- ImageDataGenerator, CNN

 Output
- Trained model: `binary_dr_cnn.keras`

