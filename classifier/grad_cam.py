import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load Pre-trained ResNet50 Model
model = ResNet50(weights="imagenet")

def get_gradcam(img_path, layer_name="conv5_block3_3_conv"):

    # Load & Preprocess Image
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get Model Outputs for Grad-CAM
    grad_model = Model(inputs=model.input, 
                       outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        tape.watch(conv_outputs)
        class_index = np.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  

    # Generate Heatmap
    heatmap = np.mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) 

    # Load Original Image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    threshold_map = np.where(heatmap<0.5, 0, 1)
    score = threshold_map.sum()/(threshold_map.shape[0]*threshold_map.shape[1])
    # Convert Heatmap to Color
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    imga = Image.fromarray(heatmap)
    imga.show()
    # Superimpose Heatmap on Image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    gradcam_path = img_path.replace(".png", "_gradcam.png")
    cv2.imwrite(gradcam_path, superimposed_img)

    print(f"Grad-CAM saved at: {gradcam_path}") 

    return (gradcam_path, score)

get_gradcam(r'F:\Sem 6\DSA\Lab\Covid19Predictor\classifier\test_images/cov.png')