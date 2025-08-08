import os
import numpy as np
import joblib
from PIL import Image 
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load the trained Random Forest model
rf_model = joblib.load("random_forest.pkl")  

# Load ResNet50 model for feature extraction (without top layer)
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path):
    
    # Open image, in RGB format
    img = Image.open(img_path).convert("RGB")
    
    # Resize image to match ResNet50 input size
    img = img.resize((224, 224))
    
    # Convert to numpy array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply preprocessing for ResNet50
    img_array = preprocess_input(img_array)
    
    # Extract features using ResNet50
    features = resnet_model.predict(img_array)
    
    return features

# Path to test image 
test_image_path = "../test_images/cov.png"  

if os.path.exists(test_image_path):
    img_features = extract_features(test_image_path)
    
    # For debugging...
    print("Extracted Features Shape (Before Reshape):", img_features.shape)
    
    
    img_features = img_features.reshape(1, -1)  
    print("Extracted Features Shape (After Reshape):", img_features.shape)

    prediction = rf_model.predict(img_features)
    
    # Check raw prediction output
    print("Raw Model Output:", prediction)

    # class mapping
    predicted_class = "COVID" if prediction[0] == 1 else "Non-COVID"
    print("Predicted Class:", predicted_class)

else:
    print(f"Error: Image file '{test_image_path}' not found.")
