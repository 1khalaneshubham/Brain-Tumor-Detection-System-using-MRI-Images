import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/brain_tumor_cnn.h5')

# Function to preprocess the image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict
def predict_tumor(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    return 'Tumor' if np.argmax(prediction) == 1 else 'No Tumor'

# Example usage
if __name__ == "__main__":
    img_path = 'data/test/tumor/1.jpg'  # Replace with your image path
    result = predict_tumor(img_path)
    print(f"Prediction: {result}")