import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the model and preprocessing objects
MODEL_PATH = os.path.join("model", "titanic_survival_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")
LE_SEX_PATH = os.path.join("model", "label_encoder_sex.pkl")
LE_EMBARKED_PATH = os.path.join("model", "label_encoder_embarked.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le_sex = joblib.load(LE_SEX_PATH)
    le_embarked = joblib.load(LE_EMBARKED_PATH)
    print("✓ Model and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.route("/")
def home():
    """Render the home page"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Make prediction based on user input"""
    try:
        # Get data from request
        data = request.get_json()

        # Extract features
        pclass = int(data["pclass"])
        sex = data["sex"]
        age = float(data["age"])
        fare = float(data["fare"])
        embarked = data["embarked"]

        # Encode categorical variables
        sex_encoded = le_sex.transform([sex])[0]
        embarked_encoded = le_embarked.transform([embarked])[0]

        # Create feature array
        features = np.array([[pclass, sex_encoded, age, fare, embarked_encoded]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Prepare response
        result = {
            "prediction": int(prediction),
            "survival_probability": float(probability[1] * 100),
            "death_probability": float(probability[0] * 100),
            "message": "Survived" if prediction == 1 else "Did Not Survive",
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
