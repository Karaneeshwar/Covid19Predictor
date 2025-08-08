import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, session
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)


from classifier.grad_cam import get_gradcam
from risk.test import predict, compute_overall

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "classifier", "random_forest.pkl")

print(f"🔍 Model Path: {MODEL_PATH}")

rf_model = joblib.load(MODEL_PATH)
print(" Random Forest model loaded successfully!")

# Load ResNet50 for feature extraction
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
print("ResNet50 model loaded for feature extraction!")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"Upload folder: {UPLOAD_FOLDER}")

def extract_features(img_path):
    try:
        print(f"Extracting features from image: {img_path}")
        img = Image.open(img_path)
        img = img.convert('RGB') 
        img = img.resize((224, 224))
        img_array = np.array(img)  
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = preprocess_input(img_array)  
        features = resnet_model.predict(img_array)
        print("Features extracted successfully!")
        return features.reshape(1, -1)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("File upload request received!")

        if "file" not in request.files:
            print("No file uploaded!")
            return render_template("index.html", error="No file uploaded!")

        file = request.files["file"]
        if file.filename == "":
            print("No file selected!")
            return render_template("index.html", error="No file selected!")

        
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print(f"Image saved at: {filepath}")

        features = extract_features(filepath)
        if features is None:
            print("Error extracting features!")
            return render_template("index.html", error="Error processing image!")

        probabilities = rf_model.predict_proba(features)
        prediction = rf_model.predict(features)[0]
        confidence = max(probabilities[0]) * 100  
        print(f"Prediction: {prediction} | Confidence: {confidence:.2f}%")

        label_map = {1: "COVID", 0: "Non-COVID"}
        predicted_label = label_map[prediction]

        # Generate Grad-CAM heatmap
        print("Generating Grad-CAM heatmap...")
        heatmap_path, sev_score = get_gradcam(filepath)  # Call function from grad_cam.py
        heatmap_filename = os.path.basename(heatmap_path)
        print(f"Heatmap saved at: {heatmap_path}")

        fields = [
            'intubated', 'pneumonia', 'age', 'sex', 'pregnancy', 'diabetes', 'copd', 'asthma',
            'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity',
            'renal_chronic', 'tobacco', 'contact_other_covid', 'icu'
        ]
        if (prediction==1):
            # Extract data from the form into a dictionary
            data = {field: request.form.get(field) for field in fields}

            df = pd.DataFrame([data])
            risk = predict(df)
            value, label = compute_overall(int(sev_score*100), risk)
            print('output from fuzzy', '\t', value, '\t', label)
            session["sev_val"] = f"{value:.2f}%"
            session["sev_label"] = label
        else:
            session['sev_val'] = None
            session['sev_score']= None
        session["image"] = file.filename
        session["prediction"] = predicted_label
        session["confidence"] = f"{confidence:.2f}%"
        session["heatmap"] = heatmap_filename
        
        return redirect(url_for("index"))

    return render_template("index.html", 
                           image=session.get("image"), 
                           prediction=session.get("prediction"), 
                           confidence=session.get("confidence"), 
                           heatmap=session.get("heatmap"),
                           sev_score=session.get("sev_val"),
                           sev_label=session.get("sev_label"))

if __name__ == "__main__":  
    print("Starting Flask App...")
    app.run(debug=True)