import numpy as np
import os


features_dir = "../features"

# Load training data
train_features = np.load(os.path.join(features_dir, "train_features.npy"))
train_labels = np.load(os.path.join(features_dir, "train_labels.npy"))

# Load test data
test_features = np.load(os.path.join(features_dir, "test_features.npy"))
test_labels = np.load(os.path.join(features_dir, "test_labels.npy"))

print("Data Loaded Successfully!")
print(f"Train Features Shape: {train_features.shape}")
print(f"Train Labels Shape: {train_labels.shape}")
print(f"Test Features Shape: {test_features.shape}")
print(f"Test Labels Shape: {test_labels.shape}")
