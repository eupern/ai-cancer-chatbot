# app.py
import streamlit as st
import os
import re
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
    "Upload medical reports or imaging files (JPG/PNG/PDF) or paste a short lab/test excerpt. "
    "Click Generate to get health summary, suggested doctor questions, and nutrition advice."
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

# ===== Helper: robust section extractor =====
def extract_section(text, header):
    """
    Extract content for a header (Summary, Questions, Nutrition) using regex with fallbacks.
    Returns stripped string or 'No findings.' if nothing found.
    """
    # Primary: look for the exact header
    pattern = rf"{header}\s*[:\-]?\s*(.*?)(?=\n(?:Summary|Questions|Nutrition)\s*[:\-]|\Z)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fallback variants
    variants = {
        "Questions": [r"questions to ask", r"doctor questions", r"questions:", r"questions to ask the doctor", r"questions for doctor"],
        "Summary": [r"summary", r"health summary", r"clinical summary"],
        "Nutrition": [r"nutrition", r"recommendations", r"diet", r"nutrition recommendations"]
    }
    for v in variants.get(header, []):
        m2 = re.search(rf"{v}\s*[:\-]?\s*(.*?)(?=\n(?:summary|questions|nutrition)\s*[:\-]|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        if m2:
            return m2.group(1).strip()

    return "No findings."

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

            # save to session_state early
            st.session_state['health_index'] = health_index

            st.subheader("📊 Health Index")
            st.write(f"Combined Health Index (0-100): {health_index}")

            # ===== GPT prompt (more strict) =====
            prompt = f"""
You are a clinical-support assistant. Given the patient's report text below, produce exactly three labelled sections: Summary, Questions, Nutrition.
- Write "Summary:" then 3-4 short sentences in plain language.
- Write "Questions:" then a numbered or bulleted list of three practical questions the patient/family should ask the doctor next visit.
- Write "Nutrition:" then three simple, food-based nutrition recommendations based on Malaysia Cancer Nutrition Guidelines.
If any section has no data to provide, write the section header and then "No findings.".

Patient report:
\"\"\"{all_lab_text}\"\"\"

Output format (must include these headers exactly): 
Summary:
- ...
Questions:
- ...
Nutrition:
- ...
"""

            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=700,
                    temperature=0.2
                )
                # robust extraction of text
                try:
                    ai_text = resp.choices[0].message.content
                except Exception:
                    ai_text = resp["choices"][0]["message"]["content"] if "choices" in resp else str(resp)

                # ===== robust section parsing =====
                summary = extract_section(ai_text, "Summary")
                questions = extract_section(ai_text, "Questions")
                nutrition = extract_section(ai_text, "Nutrition")

                # normalize tiny outputs
                if not summary or summary.strip() == "":
                    summary = "No findings."
                if not questions or questions.strip() == "":
                    questions = "No findings."
                if not nutrition or nutrition.strip() == "":
                    nutrition = "No findings."

                # save to session_state so Send button can use them reliably
                st.session_state['summary'] = summary
                st.session_state['questions'] = questions
                st.session_state['nutrition'] = nutrition
                st.session_state['ai_raw'] = ai_text

                # Display
                st.subheader("🧾 Health Summary")
                st.write(summary)
                st.subheader("❓ Suggested Questions for the Doctor")
                st.write(questions)
                st.subheader("🥗 Nutrition Recommendations")
                st.write(nutrition)

                with st.expander("Full AI output (raw)"):
                    st.code(ai_text)

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")

# ===== Button: Send WhatsApp via Twilio Sandbox with detailed logs =====
if twilio_client and st.button("Send Health Update to Family via WhatsApp"):
    if 'health_index' not in st.session_state:
        st.error("Generate AI output first before sending WhatsApp message.")
    else:
        try:
            # Use session_state values (guaranteed to exist after generation)
            message_body = (
                f"Health Index: {st.session_state.get('health_index', 'N/A')}\n\n"
                f"Summary:\n{st.session_state.get('summary', 'No findings.')}\n\n"
                f"Questions:\n{st.session_state.get('questions', 'No findings.')}\n\n"
                f"Nutrition:\n{st.session_state.get('nutrition', 'No findings.')}"
            )

            twilio_msg = twilio_client.messages.create(
                body=message_body,
                from_=st.secrets["TWILIO_WHATSAPP_FROM"],
                to=st.secrets["TWILIO_WHATSAPP_TO"]
            )
            st.success("WhatsApp message sent successfully!")
            # Detailed logs for debugging
            st.write("✅ Message SID:", getattr(twilio_msg, "sid", None))
            st.write("✅ Message Status:", getattr(twilio_msg, "status", None))
            st.write("✅ Date Created:", getattr(twilio_msg, "date_created", None))
            st.write("✅ From:", getattr(twilio_msg, "from_", None))
            st.write("✅ To:", getattr(twilio_msg, "to", None))
            # If Twilio returns errors, they may appear as attributes on the response or raise exceptions.
        except Exception as e:
            st.error(f"Failed to send WhatsApp message: {e}")
            st.write("⚠️ Check Sandbox join code, phone verification, and Twilio credentials.")






















