# ==========================================
# AI Dermatology Assistant - Streamlit App
# ==========================================

import streamlit as st
from prediction import predict_disease, predict_probability
from llm_helper import generate_ai_response


# ==========================================
# Session State
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="AI Dermatology Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/240/dermatology.png",
        width=120
    )

    st.title("AI Dermatology Assistant")

    st.markdown("---")

    st.markdown("### About Project")

    st.info(
        """
        This AI assistant predicts **10 different skin diseases**
        using a Machine Learning model trained on patient symptom data.

        After prediction, Gemini AI explains:
        - Disease overview
        - Possible causes
        - Symptoms
        - Skin care tips
        - Lifestyle recommendations
        """
    )

    st.markdown("---")

    st.success("Model Used: Tuned XGBoost")

    st.caption("Built using Python • Streamlit • XGBoost • Gemini AI")


# ==========================================
# Custom CSS Styling
# ==========================================

st.markdown("""
<style>

/* Main Background */
.stApp{
    background-color:#F4F9FF;
}

/* Main Title */
.title{
    font-size:42px;
    color:#0B5ED7;
    text-align:center;
    font-weight:bold;
}

/* Subtitle */
.subtitle{
    font-size:18px;
    text-align:center;
    color:#444444;
    margin-bottom:20px;
}

/* Section Heading */
.section{
    background-color:#EAF4FF;
    padding:12px;
    border-radius:10px;
    color:#0B5ED7;
    font-size:22px;
    font-weight:bold;
    margin-top:15px;
}

/* Prediction Card */
.result-card{
    background-color:#D1FAE5;
    padding:18px;
    border-radius:15px;
    border-left:8px solid green;
    color:black;
    font-size:20px;
}

/* Confidence Card */
.confidence-card{
    background:#FFF8E6;
    padding:15px;
    border-radius:12px;
    border-left:8px solid orange;
    color:black;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# Main Heading
# ==========================================

st.markdown(
    "<div class='title'>🩺 AI Dermatology Assistant</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict Skin Diseases using Machine Learning + Gemini AI Explanation</div>",
    unsafe_allow_html=True
)

st.markdown("---")


# ==========================================
# Welcome Banner
# ==========================================

st.markdown(
    """
    ### Welcome 👋

    Please enter your name and answer a few questions about your symptoms.
    The assistant will:

    - Predict the most likely skin disease.
    - Show prediction confidence.
    - Explain the disease using Gemini AI in simple language.
    """
)


# ==========================================
# User Information
# ==========================================

st.header("🧑 Patient Information")

username = st.text_input(
    "Enter Your Name",
    placeholder="Example: Hummad"
)

st.markdown("---")


# ==========================================
# Symptom Details
# ==========================================

st.header("🩺 Symptom Details")

st.caption("Answer the following questions about your skin condition.")


# Basic Symptoms

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

body_area = st.selectbox(
    "Affected Body Area",
    [
        "Face", "Hands", "Feet", "Scalp", "Legs",
        "Arms", "Chest", "Back", "Shoulder", "Neck"
    ]
)

itching_level = st.selectbox(
    "Itching Level",
    ["Mild", "Moderate", "Severe"]
)

rash_type = st.selectbox(
    "Rash Type",
    [
        "Pimples",
        "Circular Rash",
        "Red Rash",
        "Dry Patch",
        "Scaly Patch",
        "White Patch",
        "Blisters",
        "Bumps"
    ]
)


skin_peeling = st.radio("Skin Peeling", ["Yes", "No"], horizontal=True)
burning_sensation = st.radio("Burning Sensation", ["Yes", "No"], horizontal=True)
swelling = st.radio("Swelling", ["Yes", "No"], horizontal=True)
fever = st.radio("Fever", ["Yes", "No"], horizontal=True)
skin_color_change = st.radio("Skin Color Change", ["Yes", "No"], horizontal=True)
pain = st.radio("Pain", ["Yes", "No"], horizontal=True)
dry_skin = st.radio("Dry Skin", ["Yes", "No"], horizontal=True)
pus_or_discharge = st.radio("Pus or Discharge", ["Yes", "No"], horizontal=True)
family_history = st.radio("Family History of Skin Disease", ["Yes", "No"], horizontal=True)
allergy_history = st.radio("Allergy History", ["Yes", "No"], horizontal=True)
recent_new_product = st.radio("Used Any New Cosmetic Product Recently?", ["Yes", "No"], horizontal=True)


humidity_exposure = st.selectbox(
    "Humidity Exposure",
    ["Low", "Medium", "High"]
)

stress_level = st.selectbox(
    "Stress Level",
    ["Low", "Medium", "High"]
)

skin_type = st.selectbox(
    "Skin Type",
    ["Normal", "Dry", "Oily", "Combination", "Sensitive"]
)


st.header("📊 Health Measurements")

age = st.slider(
    "Age",
    min_value=1,
    max_value=80,
    value=25
)

bmi = st.slider(
    "BMI",
    min_value=10.0,
    max_value=40.0,
    value=22.0
)

body_temperature = st.slider(
    "Body Temperature (°F)",
    min_value=95.0,
    max_value=105.0,
    value=98.6
)

symptom_duration_days = st.slider(
    "Symptom Duration (Days)",
    min_value=1,
    max_value=60,
    value=7
)

severity_score = st.slider(
    "Severity Score",
    min_value=1,
    max_value=10,
    value=5
)

sleep_hours = st.slider(
    "Average Sleep Hours",
    min_value=3,
    max_value=12,
    value=7
)

water_intake_liters = st.slider(
    "Daily Water Intake (Liters)",
    min_value=0.5,
    max_value=5.0,
    value=2.5
)


st.markdown("---")

predict_button = st.button(
    "🔍 Predict Skin Disease",
    use_container_width=True
)


# ==========================================
# Prediction Logic
# ==========================================

if predict_button:

    # Check if user entered a name
    if username.strip() == "":
        st.warning("⚠ Please enter your name before prediction.")
        st.stop()

    else:

        # -----------------------------
        # Convert UI inputs into model inputs
        # -----------------------------

        # ==========================================
        # Raw User Input (No Encoding Required)
        # ==========================================

        user_input = {

            "gender": gender,
            "body_area": body_area,
            "itching_level": itching_level,
            "rash_type": rash_type,

            "skin_peeling": skin_peeling,
            "burning_sensation": burning_sensation,
            "swelling": swelling,
            "fever": fever,
            "skin_color_change": skin_color_change,
            "pain": pain,
            "dry_skin": dry_skin,
            "pus_or_discharge": pus_or_discharge,
            "family_history": family_history,
            "allergy_history": allergy_history,
            "recent_new_product": recent_new_product,

            "humidity_exposure": humidity_exposure,
            "stress_level": stress_level,
            "skin_type": skin_type,

            "age": age,
            "bmi": bmi,
            "body_temperature": body_temperature,
            "symptom_duration_days": symptom_duration_days,
            "severity_score": severity_score,
            "sleep_hours": sleep_hours,
            "water_intake_liters": water_intake_liters
        }
        

        # -----------------------------
        # Call Prediction Function
        # -----------------------------

        with st.spinner("Analyzing your symptoms with AI..."):

            prediction_result = predict_disease(user_input)

            predicted_disease = prediction_result["predicted_disease"]
            confidence = prediction_result["confidence"]


        # -----------------------------
        # Display Prediction
        # -----------------------------

        st.success(
            f"✅ Hello {username}! Your skin disease analysis has been completed successfully."
        )        

        st.markdown("## Prediction Result")

        st.info(f"**Predicted Disease:** {predicted_disease}")

        st.metric(
            label="Model Confidence",
            value=f"{confidence*100:.2f}%"
        )


        # Save prediction history

        st.session_state.history.append({
            "Patient": username,
            "Disease": predicted_disease,
            "Confidence": f"{confidence*100:.2f}%"
        })


        # ==========================================
        # Styled Prediction Card
        # ==========================================

        st.markdown("---")

        st.markdown(f"""
        <div style="
            background-color:#E8F8F5;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #28A745;
        ">

        <h2 style="color:#155724;">🩺 Prediction Summary</h2>

        <h3>👤 Patient: {username}</h3>

        <h3>🦠 Predicted Disease: <span style="color:#0B5ED7;">{predicted_disease}</span></h3>

        <h3>📊 Confidence Score: {confidence*100:.2f}%</h3>

        </div>
        """, unsafe_allow_html=True)


        # ==========================================
        # Confidence Meter
        # ==========================================

        st.markdown("### 📈 Model Confidence")

        st.progress(float(confidence))

        if confidence >= 0.90:
            st.success("The model is highly confident about this prediction.")

        elif confidence >= 0.75:
            st.info("The model is moderately confident about this prediction.")

        else:
            st.warning("The prediction confidence is relatively low. Consider consulting a dermatologist.")


            st.markdown("---")
        st.subheader("🎯 Top 3 Possible Skin Diseases")

        try:
            probability_df = predict_probability(user_input)

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.warning("Could not load Top 3 disease probabilities.")
            st.caption(f"Error: {e}")


        # ==========================================
        # Gemini AI Health Explanation
        # ==========================================

        st.markdown("---")
        st.subheader("🤖 AI Dermatology Assistant")

        try:
            ai_response = generate_ai_response(
                username=username,
                disease=predicted_disease,
                confidence=confidence * 100
            )

        except Exception:
            ai_response = """
            Gemini AI explanation is currently unavailable.

            Please check your internet connection or Gemini API key.
            """

        with st.expander("📖 View AI Dermatologist Report", expanded=True):
            st.markdown(ai_response)


        # ==========================================
        # Report Download Button
        # ==========================================

        st.download_button(
            label="📄 Download AI Skin Report",
            data=ai_response,
            file_name=f"{username}_skin_report.txt",
            mime="text/plain"
        )


# ==========================================
# Prediction History
# ==========================================

st.markdown("---")
st.subheader("📋 Prediction History")

import pandas as pd

history_df = pd.DataFrame(st.session_state.history)

st.dataframe(
    history_df,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# Medical Emergency Alert
# ==========================================

st.markdown("---")

st.error("""
### 🚨 Seek Immediate Medical Attention If You Experience:

- Difficulty breathing.
- Swelling of the lips, tongue, or throat.
- Rapidly spreading skin rash.
- High fever with severe skin infection.
- Persistent bleeding, pus, or worsening skin lesions.

This AI assistant is for educational purposes only.
""")


# ==========================================
# General Skin Care Tips
# ==========================================

st.markdown("---")

st.markdown("## 🌿 General Skin Care Tips")

tip1, tip2 = st.columns(2)

with tip1:
    st.success("""
    **Healthy Habits**

    - Drink sufficient water daily.
    - Wash skin gently.
    - Use sunscreen regularly.
    - Moisturize dry skin.
    - Sleep 7–8 hours every night.
    """)

with tip2:
    st.warning("""
    **Avoid**

    - Scratching affected skin.
    - Sharing towels or razors.
    - Excessive sun exposure.
    - Harsh cosmetic products.
    - Self-medication without advice.
    """)


# ==========================================
# Reset Button
# ==========================================

st.markdown("---")

if st.button("🔄 Start New Prediction"):

    st.session_state.history = []

    st.rerun()


# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:gray;">

### 🩺 AI Dermatology Assistant

Built for a **Data Science Capstone Project**

**Technologies Used**

Python | Streamlit | XGBoost | Scikit-Learn | Gemini AI

**Disclaimer**

This application provides AI-assisted educational information and should not be considered a substitute for professional medical diagnosis or treatment.

© 2026 AI Dermatology Assistant

</div>
""", unsafe_allow_html=True)


