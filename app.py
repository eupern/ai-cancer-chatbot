# app.py
import streamlit as st
import os
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr
import re
from twilio.rest import Client as TwilioClient

# ====== EasyOCR setup ======
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# ====== Streamlit config ======
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("🧠 AI-Driven Personalized Cancer Care Chatbot")
st.write(
    "Upload a medical report (JPG/PNG/PDF) or paste a short test/result. Click Generate to get a health summary, suggested doctor questions, nutrition advice, and send WhatsApp updates."
)

# ====== OpenAI API key and client ======
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Paste a key for this session (not saved).")
    api_key_input = st.text_input("OpenAI API key:", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ====== Twilio config (optional, for WhatsApp notifications) ======
TWILIO_ACCOUNT_SID = st.secrets.get("TWILIO_ACCOUNT_SID") if "TWILIO_ACCOUNT_SID" in st.secrets else os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = st.secrets.get("TWILIO_AUTH_TOKEN") if "TWILIO_AUTH_TOKEN" in st.secrets else os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = st.secrets.get("TWILIO_WHATSAPP_FROM") if "TWILIO_WHATSAPP_FROM" in st.secrets else os.getenv("TWILIO_WHATSAPP_FROM")

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

# ====== Input UI ======
st.subheader("1) Input report or summary")
uploaded_file = st.file_uploader("Upload a medical report (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])
text_input = st.text_area("Or paste a short lab summary / excerpt here", height=160)

# ====== OCR extraction ======
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
            st.warning("Uploaded file type not supported.")
    except Exception as e:
        st.error(f"OCR processing failed: {e}")

if ocr_text.strip():
    text_input = st.text_area("OCR extracted text (editable)", value=ocr_text, height=200)

input_source = text_input.strip() if text_input and text_input.strip() else None

# ====== Health Index computation ======
def compute_health_index_smart(report_text):
    score = 100
    summary = []
    text = report_text.lower()

    def extract_number(keyword):
        pattern = rf"{keyword}[:\s]*([\d\.]+)"
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except:
                return None
        return None

    # WBC
    wbc = extract_number("wbc")
    if wbc is not None:
        if wbc < 4:
            score -= 20; summary.append(f"WBC low ({wbc})")
        elif wbc > 10:
            score -= 10; summary.append(f"WBC high ({wbc})")
        else:
            summary.append(f"WBC normal ({wbc})")

    # Hb
    hb = extract_number("hb")
    if hb is not None:
        if hb < 12:
            score -= 15; summary.append(f"Hb low ({hb})")
        elif hb > 17.5:
            score -= 5; summary.append(f"Hb high ({hb})")
        else:
            summary.append(f"Hb normal ({hb})")

    # PLT
    plt = extract_number("plt")
    if plt is not None:
        if plt < 150:
            score -= 10; summary.append(f"PLT low ({plt})")
        elif plt > 450:
            score -= 10; summary.append(f"PLT high ({plt})")
        else:
            summary.append(f"PLT normal ({plt})")

    # Glucose
    glucose = extract_number("glucose")
    if glucose is not None:
        if glucose < 70:
            score -= 10; summary.append(f"Glucose low ({glucose})")
        elif glucose > 100:
            score -= 10; summary.append(f"Glucose high ({glucose})")
        else:
            summary.append(f"Glucose normal ({glucose})")

    # Blood Pressure
    bp_match = re.search(r"bp[:\s]*([\d]+)[\/\s]([\d]+)", text)
    if bp_match:
        syst, diast = int(bp_match.group(1)), int(bp_match.group(2))
        if syst > 140 or diast > 90:
            score -= 10; summary.append(f"BP high ({syst}/{diast})")
        elif syst < 90 or diast < 60:
            score -= 5; summary.append(f"BP low ({syst}/{diast})")
        else:
            summary.append(f"BP normal ({syst}/{diast})")

    score = max(0, min(100, score))
    return score, "; ".join(summary)

# ====== Generate Button ======
if st.button("Generate Summary & Recommendations"):
    if not input_source:
        st.error("Provide report text or upload OCR-compatible file.")
    elif not client:
        st.error("OpenAI client not configured.")
    else:
        with st.spinner("Processing..."):
            # Health Index
            health_index, health_summary_text = compute_health_index_smart(input_source)
            st.subheader("📊 Health Index")
            st.write(f"{health_index}/100")
            st.write(health_summary_text)

            # GPT Prompt
            prompt = f"""
You are a clinical-support assistant. Given the patient's report text below, produce:
1) A concise health summary in plain language (3-4 short sentences).
2) Three practical questions the patient/family should ask the doctor at the next visit.
3) Three personalized, food-based nutrition recommendations based on Malaysia Cancer Nutrition Guidelines (simple and actionable).

Patient report:
\"\"\"{input_source}\"\"\"

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
                        ai_text = str(resp)

                # Display GPT output
                st.subheader("🧾 Health Summary (AI)")
                if "Summary:" in ai_text:
                    try: st.write(ai_text.split("Summary:")[1].split("Questions:")[0].strip())
                    except: st.write(ai_text)
                else: st.write(ai_text[:1000] + "...")

                st.subheader("❓ Suggested Questions for the Doctor")
                if "Questions:" in ai_text:
                    try: st.write(ai_text.split("Questions:")[1].split("Nutrition:")[0].strip())
                    except: st.write("Questions parse error; see raw output below.")
                else:
                    st.write("No 'Questions:' section detected.")

                st.subheader("🥗 Nutrition Recommendations")
                if "Nutrition:" in ai_text:
                    try: st.write(ai_text.split("Nutrition:")[1].strip())
                    except: st.write("Nutrition parse error; see raw output below.")
                else:
                    st.write("No 'Nutrition:' section detected.")

                with st.expander("Full AI output (raw)"):
                    st.code(ai_text)

                # ===== Optional: Send WhatsApp =====
                if twilio_client and TWILIO_WHATSAPP_FROM:
                    st.subheader("📲 WhatsApp Notification")
                    whatsapp_number = st.text_input("Recipient WhatsApp number (+countrycode, e.g., +60123456789):")
                    if whatsapp_number and st.button("Send WhatsApp Update"):
                        message_text = f"Health Index: {health_index}/100\n{health_summary_text}\n\nAI Summary & Recommendations:\n{ai_text}"
                        try:
                            twilio_client.messages.create(
                                body=message_text,
                                from_=f"whatsapp:{TWILIO_WHATSAPP_FROM}",
                                to=f"whatsapp:{whatsapp_number}"
                            )
                            st.success("WhatsApp message sent!")
                        except Exception as e:
                            st.error(f"Failed to send WhatsApp message: {e}")
                    elif not whatsapp_number:
                        st.info("Enter a WhatsApp number above to send updates.")

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")













