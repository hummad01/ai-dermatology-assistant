import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Create Gemini client
client = genai.Client(api_key=api_key)

print("Gemini Client Loaded Successfully!")


# ============================================
# AI Dermatology Response Function
# ============================================

from google import genai

def generate_ai_response(username, disease, confidence):
    prompt = f"""
    You are an experienced dermatologist assisting patients.

    Patient Name: {username}

    The patient filled out a symptom-based dermatology assessment form.

    The machine learning model predicts **{disease}** with a confidence score of **{confidence*100:.2f}%**.

    Provide your response in simple, patient-friendly language.

    Include:
    1. What this skin condition is.
    2. Common symptoms.
    3. Possible causes or triggers.
    4. Basic home care and precautions.
    5. When to consult a dermatologist.

    End with a disclaimer that this is an AI prediction based on symptoms and is not a confirmed medical diagnosis.
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text

# ============================================
# Test Gemini Helper
# ============================================

if __name__ == "__main__":

    reply = generate_ai_response(
        username="Hummad",
        disease="Psoriasis",
        confidence=92.0
    )

    print(reply)