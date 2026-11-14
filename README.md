# AI Dermatology Assistant (Capstone Project)

## Project Overview
AI Dermatology Assistant predicts one of 10 skin diseases using patient symptom data and explains the prediction using Gemini AI.

### Technologies Used
- Python
- Streamlit
- Scikit-learn
- XGBoost
- Pandas & NumPy
- Matplotlib & Seaborn
- Google Gemini API

---

## Project Structure

- data/ → Dataset files
- notebook/ → Data Cleaning, EDA and Model Training notebooks
- models/ → Saved ML models and encoders
- streamlit_app/ → Streamlit application
- reports/ → Capstone Presentation

---

## Steps to Run the Project

### Step 1: Create Virtual Environment

Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Add Gemini API Key

Create a file inside `streamlit_app` named `.env`

```env
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

### Step 4: Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

The application opens in your browser.

---

## Model Used

Best Model: Tuned XGBoost Classifier

Accuracy: **91.76%**

Supports prediction of:
- Acne
- Eczema
- Psoriasis
- Ringworm
- Rosacea
- Vitiligo
- Chickenpox
- Urticaria
- Contact Dermatitis
- Fungal Infection

---

## Note

This project was developed for academic purposes using a synthetic healthcare dataset.