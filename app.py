# app.py
import streamlit as st
import os
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr
from twilio.rest import Client as TwilioClient

# ===== EasyOCR setup =====
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# ===== Streamlit page config =====
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("🧠 AI-Driven Personalized Cancer Care Chatbot")
st.write("Upload a medical report (JPG/PNG/PDF) or paste a short test/result. Click Generate to get a health summary, suggested doctor questions, nutrition advice, and send WhatsApp notifications to family.")

# ===== OpenAI API key =====
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    api_key_input = st.text_input("Paste your OpenAI API key for this session (not saved):", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ===== Twilio WhatsApp setup =====
TWILIO_SID = st.secrets.get("TWILIO_ACCOUNT_SID") or os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = st.secrets.get("TWILIO_AUTH_TOKEN") or os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = st.secrets.get("TWILIO_WHATSAPP_FROM") or os.getenv("TWILIO_WHATSAPP_FROM")
TO_NUMBERS = st.secrets.get("TO_NUMBERS") or os.getenv("TO_NUMBERS")  # comma-separated WhatsApp numbers

twilio_client = None
if TWILIO_SID and TWILIO_TOKEN:
    twilio_client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)

# ===== Input UI =====
st.subheader("1) Input report or summary")
uploaded_file = st.file_uploader("Upload a medical report (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])
text_input = st.text_area("Or paste a short lab summary / excerpt here", height=160)

# ===== OCR extraction =====
ocr_text = ""
if uploaded_file:
    try:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded report preview", use_column_width=True)
            result = reader.readtext(np.array(image), detail=0)
            ocr_text = "\n".join(result)
        elif uploaded_file.type == "application/pdf":
            pages = convert_from_bytes(uploaded_file.read())
            for page in pages:
                page_arr = np.array(page)
                result = reader.readtext(page_arr, detail=0)
                ocr_text += "\n".join(result) + "\n"
        else:
            st.warning("Unsupported file type for OCR.")
    except Exception as e:
        st.error(f"OCR processing failed: {e}")

if ocr_text.strip():
    text_input = st.text_area("OCR extracted text (editable)", value=ocr_text, height=200)

input_source = text_input.strip() if text_input and text_input.strip() else None

# ===== Helper: Health Index calculation (simple example) =====
def calculate_health_index(text):
    # Simple placeholder: more mentions of "normal" => higher score
    score = 50
    text_lower = text.lower()
    if "normal" in text_lower:
        score += 30
    if "high" in text_lower or "low" in text_lower:
        score -= 20
    return max(0, min(100, score))

# ===== Generate button =====
if st.button("Generate Summary, Nutrition & Notify Family"):
    if not input_source:
        st.error("Please provide report text or upload OCR-compatible file first.")
    elif not client:
        st.error("OpenAI client not configured.")
    else:
        with st.spinner("Generating AI output..."):
            # Prompt GPT
            prompt = f"""
You are a clinical-support assistant. Given the patient's report text below, produce:
1) A concise health summary in plain language (3-4 short sentences).
2) Three practical questions the patient/family should ask the doctor at the next visit.
3) Three personalized, food-based nutrition recommendations based on Malaysia Cancer Nutrition Guidelines (simple and actionable).

Patient report:
\"\"\"{input_source}\"\"\"

Output format:
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
                    max_tokens=500,
                    temperature=0.2
                )

                try:
                    ai_text = resp.choices[0].message.content
                except:
                    ai_text = resp["choices"][0]["message"]["content"]

                # ===== Display Health Index =====
                health_index = calculate_health_index(input_source)
                st.subheader("📊 Health Index")
                st.progress(health_index / 100)
                st.write(f"Health Index Score: {health_index}/100")

                # ===== Display GPT output sections =====
                st.subheader("🧾 Health Summary")
                summary = ai_text.split("Summary:")[1].split("Questions:")[0].strip() if "Summary:" in ai_text else ai_text
                st.write(summary)

                st.subheader("❓ Suggested Questions for the Doctor")
                questions = ai_text.split("Questions:")[1].split("Nutrition:")[0].strip() if "Questions:" in ai_text else "See summary above."
                st.write(questions)

                st.subheader("🥗 Nutrition Recommendations")
                nutrition = ai_text.split("Nutrition:")[1].strip() if "Nutrition:" in ai_text else "No nutrition section detected."
                st.write(nutrition)

                # ===== WhatsApp Notification =====
                if twilio_client and TWILIO_FROM and TO_NUMBERS:
                    message_text = f"Health Update:\n{summary}\n\nDoctor Questions:\n{questions}\n\nNutrition:\n{nutrition}\n\nHealth Index: {health_index}/100"
                    for number in TO_NUMBERS.split(","):
                        number = number.strip()
                        try:
                            twilio_client.messages.create(
                                from_=TWILIO_FROM,
                                to=f"whatsapp:{number}",
                                body=message_text
                            )
                        except Exception as e:
                            st.warning(f"Failed to send WhatsApp message to {number}: {e}")
                    st.success("WhatsApp notifications sent to family members (if configured).")
                else:
                    st.info("WhatsApp not configured. Set Twilio credentials in Secrets or environment variables to enable notifications.")

                # ===== Expandable raw AI output =====
                with st.expander("Full AI output (raw)"):
                    st.code(ai_text)

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")












