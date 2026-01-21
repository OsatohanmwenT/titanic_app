"""
Titanic Survival Prediction - Model Development Script
This script trains the Random Forest model and saves it for use in the web application.
Run this script before starting the Flask app.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    print("=" * 60)
    print("TITANIC SURVIVAL PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)

    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    print(f"   ✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Select features
    print("\n2. Selecting features...")
    selected_features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Survived"]
    df_selected = df[selected_features].copy()
    print("   ✓ Selected 5 features + target variable")

    # Handle missing values
    print("\n3. Handling missing values...")
    df_selected["Age"].fillna(df_selected["Age"].median(), inplace=True)
    df_selected["Embarked"].fillna(df_selected["Embarked"].mode()[0], inplace=True)
    df_selected["Fare"].fillna(df_selected["Fare"].median(), inplace=True)
    print("   ✓ Missing values handled")

    # Encode categorical variables
    print("\n4. Encoding categorical variables...")
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df_selected["Sex"] = le_sex.fit_transform(df_selected["Sex"])
    df_selected["Embarked"] = le_embarked.fit_transform(df_selected["Embarked"])
    print("   ✓ Categorical variables encoded")

    # Prepare features and target
    print("\n5. Preparing features and target...")
    X = df_selected.drop("Survived", axis=1)
    y = df_selected["Survived"]
    print(f"   ✓ Features: {X.shape}, Target: {y.shape}")

    # Feature scaling
    print("\n6. Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("   ✓ Features scaled")

    # Train-test split
    print("\n7. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"   ✓ Training: {X_train.shape[0]} samples, Testing: {X_test.shape[0]} samples"
    )

    # Train model
    print("\n8. Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    print("   ✓ Model trained successfully")

    # Evaluate model
    print("\n9. Evaluating model...")
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\n   Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"   Testing Accuracy: {test_accuracy*100:.2f}%")

    print("\n   Classification Report:")
    print(
        classification_report(
            y_test, y_test_pred, target_names=["Did Not Survive", "Survived"]
        )
    )

    # Save model and preprocessing objects
    print("\n10. Saving model and preprocessing objects...")
    joblib.dump(rf_model, "model/titanic_survival_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(le_sex, "model/label_encoder_sex.pkl")
    joblib.dump(le_embarked, "model/label_encoder_embarked.pkl")

    print("   ✓ Model saved to: model/titanic_survival_model.pkl")
    print("   ✓ Scaler saved to: model/scaler.pkl")
    print("   ✓ Label encoders saved")

    # Test model reload
    print("\n11. Testing model reload...")
    loaded_model = joblib.load("model/titanic_survival_model.pkl")
    loaded_scaler = joblib.load("model/scaler.pkl")

    # Make a test prediction
    test_data = np.array(
        [[1, 0, 29, 100, 0]]
    )  # 1st class, female, age 29, fare 100, embarked C
    test_scaled = loaded_scaler.transform(test_data)
    prediction = loaded_model.predict(test_scaled)[0]
    probability = loaded_model.predict_proba(test_scaled)[0]

    print("   ✓ Model reloaded successfully")
    print(
        f"   ✓ Test prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}"
    )
    print(f"   ✓ Survival probability: {probability[1]*100:.2f}%")

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nYou can now run the Flask application with: python app.py")


if __name__ == "__main__":
    main()
