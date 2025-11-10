# app.py
import streamlit as st
import os
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr
from twilio.rest import Client as TwilioClient

# ===== Initialize EasyOCR reader =====
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)  # GPU disabled for Streamlit Cloud

# ===== Streamlit page config =====
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("🧠 AI-Driven Personalized Cancer Care Chatbot")
st.write(
    "Upload medical reports or imaging files (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get health summary, suggested doctor questions, and nutrition advice."
)

# ===== OpenAI API key and client setup =====
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found in Streamlit Secrets. Paste below for this session:")
    api_key_input = st.text_input("OpenAI API Key:", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ===== Twilio Setup (Optional) =====
twilio_client = None
if "TWILIO_ACCOUNT_SID" in st.secrets and "TWILIO_AUTH_TOKEN" in st.secrets:
    twilio_client = TwilioClient(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])

# ===== Input UI =====
st.subheader("1) Input medical reports or lab summary")
uploaded_files = st.file_uploader(
    "Upload medical reports / imaging files (JPG/PNG/PDF). You can upload multiple files.",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)
text_input = st.text_area("Or paste a short lab/test excerpt here", height=160)

# ===== OCR Extraction =====
lab_texts = []
image_texts = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type.startswith("image"):
                img = Image.open(uploaded_file)
                st.image(img, caption=f"Preview: {uploaded_file.name}", use_column_width=True)
                ocr_result = "\n".join(reader.readtext(np.array(img), detail=0))
            elif uploaded_file.type == "application/pdf":
                pages = convert_from_bytes(uploaded_file.read())
                ocr_result = ""
                for page in pages:
                    ocr_result += "\n".join(reader.readtext(np.array(page), detail=0)) + "\n"
            else:
                st.warning(f"{uploaded_file.name} is not a supported file type.")
                continue

            fname_lower = uploaded_file.name.lower()
            if any(k in fname_lower for k in ["pet", "ct", "xray", "scan"]):
                image_texts.append(ocr_result)
            else:
                lab_texts.append(ocr_result)

        except Exception as e:
            st.error(f"OCR failed for {uploaded_file.name}: {e}")

# If OCR found text, show editable text area
if lab_texts:
    text_input = st.text_area("OCR extracted lab text (editable)", value="\n".join(lab_texts), height=200)

input_source = text_input.strip() if text_input and text_input.strip() else None

# ===== Health Index Calculation =====
def compute_health_index_smart(report_text):
    score = 50
    keywords_positive = ["normal", "stable", "remission", "improved"]
    keywords_negative = ["high", "low", "elevated", "decreased", "critical", "abnormal"]
    text_lower = report_text.lower()
    for kw in keywords_positive:
        if kw in text_lower:
            score += 5
    for kw in keywords_negative:
        if kw in text_lower:
            score -= 5
    return max(0, min(100, score))

def compute_health_index_with_imaging(report_texts, image_reports_texts=None):
    lab_score = compute_health_index_smart(report_texts) if report_texts else 50
    image_score = 100
    if image_reports_texts:
        deductions = []
        for text in image_reports_texts:
            text_lower = text.lower()
            for kw in ["metastasis", "lesion", "tumor growth", "progression"]:
                if kw in text_lower:
                    deductions.append(10)
            for kw in ["stable", "no abnormality", "remission"]:
                if kw in text_lower:
                    deductions.append(-5)
        total_deduction = sum(deductions)
        image_score = max(0, min(100, image_score - total_deduction))
    combined_score = lab_score * 0.7 + image_score * 0.3
    return round(combined_score, 1)

# ===== Button: Generate AI output =====
if st.button("Generate Summary & Recommendations"):
    if not input_source and not lab_texts:
        st.error("Please paste a lab/test excerpt or upload OCR-compatible files first.")
    elif not client:
        st.error("OpenAI client not configured. Please set OPENAI_API_KEY in Streamlit Secrets or paste above.")
    else:
        with st.spinner("Generating AI output..."):
            all_lab_text = input_source if input_source else "\n".join(lab_texts)
            health_index = compute_health_index_with_imaging(all_lab_text, image_texts)
            st.subheader("📊 Health Index")
            st.write(f"Combined Health Index (0-100): {health_index}")

            # GPT Prompt
            prompt = f"""
You are a clinical-support assistant. Given the patient's report text below, produce:
1) A concise health summary in plain language (3-4 short sentences).
2) Three practical questions the patient/family should ask the doctor at the next visit.
3) Three personalized, food-based nutrition recommendations based on Malaysia Cancer Nutrition Guidelines (simple and actionable).

Patient report:
\"\"\"{all_lab_text}\"\"\"

Output format:
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
                # Extract text robustly
                try:
                    ai_text = resp.choices[0].message.content
                except Exception:
                    ai_text = resp["choices"][0]["message"]["content"] if "choices" in resp else str(resp)

                # Display sections
                st.subheader("🧾 Health Summary")
                if "Summary:" in ai_text:
                    summary = ai_text.split("Summary:")[1].split("Questions:")[0].strip()
                    st.write(summary)
                else:
                    summary = ai_text
                    st.write(summary)

                st.subheader("❓ Suggested Questions for the Doctor")
                if "Questions:" in ai_text:
                    questions = ai_text.split("Questions:")[1].split("Nutrition:")[0].strip()
                    st.write(questions)
                else:
                    st.write("No clearly labeled 'Questions' section detected.")

                st.subheader("🥗 Nutrition Recommendations")
                if "Nutrition:" in ai_text:
                    nutrition = ai_text.split("Nutrition:")[1].strip()
                    st.write(nutrition)
                else:
                    st.write("No clearly labeled 'Nutrition' section detected.")

                # Optional Twilio WhatsApp send with detailed logs
                if twilio_client and st.button("Send Health Update to Family via WhatsApp"):
                    try:
                        message_body = f"Health Index: {health_index}\n\nSummary:\n{summary}\n\nNutrition:\n{nutrition}"
                        twilio_msg = twilio_client.messages.create(
                            body=message_body,
                            from_=st.secrets["TWILIO_WHATSAPP_FROM"],
                            to=st.secrets["TWILIO_WHATSAPP_TO"]
                        )
                        st.success("WhatsApp message sent successfully!")
                        st.write("✅ Message SID:", twilio_msg.sid)
                        st.write("✅ Message Status:", twilio_msg.status)
                        st.write("✅ Date Created:", twilio_msg.date_created)
                        st.write("✅ From:", twilio_msg.from_)
                        st.write("✅ To:", twilio_msg.to)
                    except Exception as e:
                        st.error(f"Failed to send WhatsApp message: {e}")
                        st.write("⚠️ Check Sandbox join code, phone verification, and Twilio credentials.")

                with st.expander("Full AI output (raw)"):
                    st.code(ai_text)

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")





















