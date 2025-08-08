import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib


train_features = np.load("../features/train_features.npy")
train_labels = np.load("../features/train_labels.npy")
test_features = np.load("../features/test_features.npy")
test_labels = np.load("../features/test_labels.npy")


xgb_model = xgb.XGBClassifier(
    n_estimators=100,   # Number of trees
    learning_rate=0.1,  # Step size shrinkage
    max_depth=5,        # Depth of trees
    use_label_encoder=False,  
    eval_metric="logloss"     
)


xgb_model.fit(train_features, train_labels)


predictions = xgb_model.predict(test_features)

accuracy = accuracy_score(test_labels, predictions)
print(f"XGBoost Accuracy: {accuracy:.4f}")


model_path = "./xgboost_model.pkl"
joblib.dump(xgb_model, model_path)
print(f"Model saved successfully at: {model_path}")
