import os
import pickle
import numpy as np
from features import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model", "cleanspot_model.pkl")

_model = None

def load_model():
    global _model

    if _model is None:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                "ML model not found. Add dataset images and run train_model.py first."
            )

        with open(MODEL_FILE, "rb") as file:
            _model = pickle.load(file)

    return _model

def predict_waste(image_path):
    model = load_model()
    features = extract_features(image_path).reshape(1, -1)

    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
    else:
        confidence = 0.0

    return str(prediction), confidence
