# app.py
import streamlit as st
import os
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr
import re

# Initialize EasyOCR reader (Simplified Chinese + English)
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# Streamlit page config
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")

st.title("🧠 AI-Driven Personalized Cancer Care Chatbot")
st.write(
    "Upload a medical report (JPG/PNG/PDF) or paste a short test/result. Click Generate to get a health summary, suggested doctor questions, nutrition advice, and a health index."
)

# ===== OpenAI API key and client setup =====
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Add OPENAI_API_KEY in Secrets or paste a key below for a quick test.")
    api_key_input = st.text_input("Paste your OpenAI API key for this session (not saved):", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ===== Input UI =====
st.subheader("1) Input report or summary")
uploaded_file = st.file_uploader("Upload a medical report (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])
text_input = st.text_area("Or paste a short lab summary / excerpt here", height=160)

# ===== OCR extraction (EasyOCR) =====
ocr_text = ""
if uploaded_file:
    try:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded report (preview)", use_column_width=True)
            result = reader.readtext(np.array(image), detail=0)
            ocr_text = "\n".join(result)
        elif uploaded_file.type == "application/pdf":
            pages = convert_from_bytes(uploaded_file.read())
            for page in pages:
                page_arr = np.array(page)
                result = reader.readtext(page_arr, detail=0)
                ocr_text += "\n".join(result) + "\n"
        else:
            st.warning("Uploaded file type is not supported for OCR.")
    except Exception as e:
        st.error(f"OCR processing failed: {e}")

# If OCR found text, present it in an editable text area
if ocr_text.strip():
    text_input = st.text_area("OCR extracted text (editable)", value=ocr_text, height=200)

input_source = text_input.strip() if text_input and text_input.strip() else None

# ===== Health Index calculation function =====
def calculate_health_index(report_text):
    """
    Extracts some common lab values and computes a simple health score 0-100.
    This is an example; you can extend it with more indicators.
    """
    score = 0
    indicators = {
        "Glucose": (70, 140),     # normal range
        "WBC": (4, 10),           # x10^9/L
        "Hb": (12, 16),           # g/dL
        "Platelets": (150, 400)   # x10^9/L
    }
    found_values = {}
    for key, (low, high) in indicators.items():
        # Simple regex to find numbers after indicator names
        match = re.search(rf"{key}[:\s]*([\d\.]+)", report_text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            found_values[key] = val
            # linear scaling: 0 if below low, 10 if above high, proportionally in between
            if val < low:
                score += 0
            elif val > high:
                score += 10
            else:
                score += int(10 * (val - low) / (high - low))
        else:
            # missing indicator: assign neutral 5
            score += 5
    # Average across indicators
    health_score = int(score / len(indicators) * 10)  # scale to 0-100
    if health_score >= 80:
        status = "Good"
    elif health_score >= 50:
        status = "Moderate"
    else:
        status = "Low"
    return health_score, status, found_values

# ===== Button: call OpenAI model =====
if st.button("Generate Summary, Health Index & Recommendations"):
    if not input_source:
        st.error("Please paste a short report excerpt or upload an OCR-compatible file first.")
    elif not client:
        st.error("OpenAI client not configured. Please set OPENAI_API_KEY in Streamlit Secrets or paste it above.")
    else:
        with st.spinner("Calculating health index..."):
            health_score, health_status, found_values = calculate_health_index(input_source)
            st.subheader("📊 Health Index")
            st.metric(label="Overall Health Score", value=f"{health_score}/100", delta=health_status)
            st.write("Detected lab values:", found_values)

        with st.spinner("Generating AI output..."):
            prompt = f"""
You are a clinical-support assistant. Given the patient's report text below, produce:
1) A concise health summary in plain language (3-4 short sentences).
2) Three practical questions the patient/family should ask the doctor at the next visit.
3) Three personalized, food-based nutrition recommendations based on Malaysia Cancer Nutrition Guidelines (simple and actionable).

Patient report:
\"\"\"{input_source}\"\"\"

Health index: {health_score}/100

Output format (use these labels):
Summary:
- ...
Questions:
- ...
Nutrition:
- ...
Keep the language simple and suitable for elderly patients and family members.
"""

            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.2
                )

                ai_text = ""
                try:
                    ai_text = resp.choices[0].message.content
                except Exception:
                    try:
                        ai_text = resp["choices"][0]["message"]["content"]
                    except Exception:
                        try:
                            ai_text = resp.choices[0].text
                        except Exception:
                            ai_text = str(resp)

                # ===== Display parsed sections =====
                st.subheader("🧾 Health Summary")
                if "Summary:" in ai_text:
                    try:
                        summary = ai_text.split("Summary:")[1].split("Questions:")[0].strip()
                        st.write(summary)
                    except Exception:
                        st.write(ai_text)
                else:
                    st.write(ai_text if len(ai_text) < 1000 else ai_text[:1000] + "...")

                st.subheader("❓ Suggested Questions for the Doctor")
                if "Questions:" in ai_text:
                    try:
                        questions = ai_text.split("Questions:")[1].split("Nutrition:")[0].strip()
                        st.write(questions)
                    except Exception:
                        st.write("Questions section parse error; see full output below.")
                else:
                    st.write("No clearly labeled 'Questions:' section detected. S









