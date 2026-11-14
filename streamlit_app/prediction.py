# ======================================
# Skin Disease Prediction Module
# ======================================

# This file loads the trained machine learning model
# and predicts the disease based on user symptoms.

import joblib
import pandas as pd


# ======================================
# Load Saved Files
# ======================================

from pathlib import Path

# Path of prediction.py file
BASE_DIR = Path(__file__).resolve().parent

# Path to models folder
MODEL_DIR = BASE_DIR.parent / "models"

# Load saved files
model = joblib.load(MODEL_DIR / "dermatology_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
encoder = joblib.load(MODEL_DIR / "disease_encoder.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")


# ======================================
# Prediction Function
# ======================================


def predict_disease(user_input):
    """
    Predicts skin disease from raw user input.
    Handles encoding, scaling and prediction internally.
    """

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([user_input])

    # ==============================
    # One-Hot Encoding
    # ==============================

    input_df["gender_Male"] = (input_df["gender"] == "Male").astype(int)
    input_df["gender_Other"] = (input_df["gender"] == "Other").astype(int)

    input_df["skin_type_Dry"] = (input_df["skin_type"] == "Dry").astype(int)
    input_df["skin_type_Normal"] = (input_df["skin_type"] == "Normal").astype(int)
    input_df["skin_type_Oily"] = (input_df["skin_type"] == "Oily").astype(int)
    input_df["skin_type_Sensitive"] = (input_df["skin_type"] == "Sensitive").astype(int)

    # Remove original columns
    input_df.drop(columns=["gender", "skin_type"], inplace=True)

    # ==============================
    # Label Encode Remaining Columns
    # ==============================

    # ==============================
    # Manual Encoding
    # ==============================

    # body_area
    body_area_map = {
        "Face": 0,
        "Hands": 1,
        "Feet": 2,
        "Scalp": 3,
        "Legs": 4,
        "Arms": 5,
        "Chest": 6,
        "Back": 7,
        "Shoulder": 8,
        "Neck": 9
    }

    # rash_type
    rash_type_map = {
        "Pimples": 0,
        "Circular Rash": 1,
        "Red Rash": 2,
        "Dry Patch": 3,
        "Scaly Patch": 4,
        "White Patch": 5,
        "Blisters": 6,
        "Bumps": 7
    }

    # Ordinal encoded columns
    itching_map = {
        "Mild": 0,
        "Moderate": 1,
        "Severe": 2
    }

    humidity_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    stress_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    input_df["body_area"] = input_df["body_area"].map(body_area_map)
    input_df["rash_type"] = input_df["rash_type"].map(rash_type_map)
    input_df["itching_level"] = input_df["itching_level"].map(itching_map)
    input_df["humidity_exposure"] = input_df["humidity_exposure"].map(humidity_map)
    input_df["stress_level"] = input_df["stress_level"].map(stress_map)

    # ==============================
    # Convert Yes/No into 1/0
    # ==============================

    binary_columns = [
        "skin_peeling",
        "burning_sensation",
        "swelling",
        "fever",
        "skin_color_change",
        "pain",
        "dry_skin",
        "pus_or_discharge",
        "family_history",
        "allergy_history",
        "recent_new_product"
    ]

    for col in binary_columns:
        input_df[col] = input_df[col].map({"Yes":1, "No":0})

    # ==============================
    # Arrange Columns
    # ==============================

    input_df = input_df[feature_names]

    # ==============================
    # Scale Numerical Features
    # ==============================

    numerical_cols = [
        "age",
        "bmi",
        "body_temperature",
        "symptom_duration_days",
        "severity_score",
        "sleep_hours",
        "water_intake_liters"
    ]

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # ==============================
    # Prediction
    # ==============================

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df).max()

    disease = encoder.inverse_transform(prediction)[0]

    return {
        "predicted_disease": disease,
        "confidence": float(probability)
    }


# ======================================
# Prediction Probability Function
# ======================================

def predict_probability(user_input):

    input_df = pd.DataFrame([user_input])

    # One-hot encoding
    input_df["gender_Male"] = (input_df["gender"] == "Male").astype(int)
    input_df["gender_Other"] = (input_df["gender"] == "Other").astype(int)

    input_df["skin_type_Dry"] = (input_df["skin_type"] == "Dry").astype(int)
    input_df["skin_type_Normal"] = (input_df["skin_type"] == "Normal").astype(int)
    input_df["skin_type_Oily"] = (input_df["skin_type"] == "Oily").astype(int)
    input_df["skin_type_Sensitive"] = (input_df["skin_type"] == "Sensitive").astype(int)

    input_df.drop(columns=["gender", "skin_type"], inplace=True)

    # Label encoding
    # ==============================
    # Manual Encoding
    # ==============================

    # body_area
    body_area_map = {
        "Face": 0,
        "Hands": 1,
        "Feet": 2,
        "Scalp": 3,
        "Legs": 4,
        "Arms": 5,
        "Chest": 6,
        "Back": 7,
        "Shoulder": 8,
        "Neck": 9
    }

    # rash_type
    rash_type_map = {
        "Pimples": 0,
        "Circular Rash": 1,
        "Red Rash": 2,
        "Dry Patch": 3,
        "Scaly Patch": 4,
        "White Patch": 5,
        "Blisters": 6,
        "Bumps": 7
    }

    # Ordinal encoded columns
    itching_map = {
        "Mild": 0,
        "Moderate": 1,
        "Severe": 2
    }

    humidity_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    stress_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    input_df["body_area"] = input_df["body_area"].map(body_area_map)
    input_df["rash_type"] = input_df["rash_type"].map(rash_type_map)
    input_df["itching_level"] = input_df["itching_level"].map(itching_map)
    input_df["humidity_exposure"] = input_df["humidity_exposure"].map(humidity_map)
    input_df["stress_level"] = input_df["stress_level"].map(stress_map)

    # Binary encoding
    binary_columns = [
        "skin_peeling",
        "burning_sensation",
        "swelling",
        "fever",
        "skin_color_change",
        "pain",
        "dry_skin",
        "pus_or_discharge",
        "family_history",
        "allergy_history",
        "recent_new_product"
    ]

    for col in binary_columns:
        input_df[col] = input_df[col].map({"Yes":1, "No":0})

    # Arrange columns
    input_df = input_df[feature_names]

    # Scale numerical features
    numerical_cols = [
        "age",
        "bmi",
        "body_temperature",
        "symptom_duration_days",
        "severity_score",
        "sleep_hours",
        "water_intake_liters"
    ]

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Prediction probabilities
    probabilities = model.predict_proba(input_df)[0]

    disease_names = encoder.classes_

    probability_df = pd.DataFrame({
        "Disease": disease_names,
        "Probability": probabilities
    })

    probability_df = probability_df.sort_values(
        by="Probability",
        ascending=False
    ).head(3)

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    ).round(2)

    return probability_df


if __name__ == "__main__":

    sample_input = {
        # We'll add all features later from app.py.
    }

    print("Prediction module loaded successfully!")