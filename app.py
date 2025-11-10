# app.py
import streamlit as st
import os
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr
from twilio.rest import Client

# ===== Initialize EasyOCR reader =====
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# ===== Streamlit page config =====
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("🧠 AI-Driven Personalized Cancer Care Chatbot")
st.write(
    "Upload medical reports (JPG/PNG/PDF, multi-page supported) or paste a short test/result. Click Generate to get a health summary, suggested doctor questions, and nutrition advice."
)

# ===== OpenAI API key & client setup =====
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Paste for this session (not saved).")
    api_key_input = st.text_input("Paste OpenAI API key:", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ===== Twilio setup =====
TWILIO_ACCOUNT_SID = st.secrets.get("TWILIO_ACCOUNT_SID", None)
TWILIO_AUTH_TOKEN = st.secrets.get("TWILIO_AUTH_TOKEN", None)
TWILIO_WHATSAPP_FROM = st.secrets.get("TWILIO_WHATSAPP_FROM", None)  # e.g., "whatsapp:+1415xxxxxxx"
WHATSAPP_TO = st.text_input("Recipient WhatsApp number (include 'whatsapp:+countrycode')", "")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

# ===== Input UI =====
st.subheader("1) Input reports or summary")
uploaded_files = st.file_uploader("Upload medical reports (JPG/PNG/PDF, multi-file supported)", type=["jpg","jpeg","png","pdf"], accept_multiple_files=True)
text_input = st.text_area("Or paste a short lab summary / excerpt here", height=160)

# ===== OCR extraction =====
ocr_text = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type.startswith("image"):
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Preview: {uploaded_file.name}", use_column_width=True)
                result = reader.readtext(np.array(image), detail=0)
                ocr_text += "\n".join(result) + "\n"
            elif uploaded_file.type == "application/pdf":
                pages = convert_from_bytes(uploaded_file.read())
                for page in pages:
                    page_arr = np.array(page)
                    result = reader.readtext(page_arr, detail=0)
                    ocr_text += "\n".join(result) + "\n"
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
        except Exception as e:
            st.error(f"OCR failed for {uploaded_file.name}: {e}")

if ocr_text.strip():
    text_input = st.text_area("OCR extracted text (editable)", value=ocr_text, height=200)

input_source = text_input.strip() if text_input and text_input.strip() else None

# ===== Health Index calculation =====
def compute_health_index(report_text):
    """
    Example heuristic-based Health Index:
    - Checks for key markers like hemoglobin, WBC, platelets, blood sugar
    - Returns 0-100 score (simplified for demonstration)
    """
    if not report_text:
        return None
    score = 100
    keywords = {
        'hemoglobin': 5,
        'wbc': 5,
        'platelets': 5,
        'glucose': 5,
        'blood sugar':5
    }
    for kw, deduction in keywords.items():
        if kw.lower() in report_text.lower():
            score -= deduction
    score = max(0, min(score,100))
    return score

health_index = compute_health_index(input_source) if input_source else None
if health_index is not None:
    st.metric("🏥 Health Index", f"{health_index}/100")

# ===== Button: Generate GPT output =====
if st.button("Generate Summary & Recommendations"):
    if not input_source:
        st.error("Please paste or upload reports first.")
    elif not client:
        st.error("OpenAI client not configured.")
    else:
        with st.spinner("Generating AI output..."):
            prompt = f"""
You are a clinical-support assistant. Based on the patient's report below, produce:
1) A concise health summary in plain language (3-4 short sentences)
2) Three practical questions for the next doctor visit
3) Three personalized nutrition recommendations (Malaysia Cancer Nutrition Guidelines)

Patient report:
\"\"\"{input_source}\"\"\"

Output format:
Summary:
- ...
Questions:
- ...
Nutrition:
- ...
Keep it simple for elderly patients/family.
Health Index: {health_index}/100
"""
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=700,
                    temperature=0.2
                )
                try:
                    ai_text = resp.choices[0].message.content
                except:
                    ai_text = resp["choices"][0]["message"]["content"]

                # ===== Display sections =====
                st.subheader("🧾 Health Summary")
                summary = ai_text.split("Summary:")[1].split("Questions:")[0].strip() if "Summary:" in ai_text else ai_text
                st.write(summary)

                st.subheader("❓ Suggested Questions for the Doctor")
                questions = ai_text.split("Questions:")[1].split("Nutrition:")[0].strip() if "Questions:" in ai_text else "See summary above."
                st.write(questions)

                st.subheader("🥗 Nutrition Recommendations")
                nutrition = ai_text.split("Nutrition:")[1].strip() if "Nutrition:" in ai_text else "No nutrition section detected."
                st.write(nutrition)

                # ===== Send WhatsApp if configured =====
                if twilio_client and WHATSAPP_TO:
                    message_text = f"Health Index: {health_index}/100\n\nSummary:\n{summary}\n\nQuestions:\n{questions}\n\nNutrition:\n{nutrition}"
                    try:
                        twilio_client.messages.create(
                            body=message_text,
                            from_=TWILIO_WHATSAPP_FROM,
                            to=WHATSAPP_TO
                        )
                        st.success("WhatsApp notification sent successfully!")
                    except Exception as e:
                        st.error(f"WhatsApp sending failed: {e}")

                # ===== Expandable raw output =====
                with st.expander("Full AI output (raw)"):
                    st.code(ai_text)

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")














