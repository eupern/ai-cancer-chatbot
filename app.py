# app.py (patched, Twilio removed, email optional added)
import streamlit as st
import os
import re
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr
import smtplib
from email.message import EmailMessage

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)

# Streamlit config
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("AI-Driven Personalized Cancer Care Chatbot")
st.write("Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, and dietitian-level nutrition advice.")

# OpenAI client setup
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    api_key_input = st.text_input("OpenAI API Key (session only):", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Input section
uploaded_files = st.file_uploader("Upload medical reports / imaging files (JPG/PNG/PDF). You can upload multiple files.", type=["jpg","jpeg","png","pdf"], accept_multiple_files=True)
text_input = st.text_area("Or paste a short lab/test excerpt here (English preferred)", height=160)

# OCR extraction
lab_texts, image_texts = [], []
if uploaded_files:
    for f in uploaded_files:
        try:
            if f.type.startswith("image"):
                img = Image.open(f).convert("RGB")
                st.image(img, caption=f"Preview: {f.name}", use_column_width=True)
                ocr_result = "\n".join(reader.readtext(np.array(img), detail=0))
            elif f.type == "application/pdf":
                pages = convert_from_bytes(f.read())
                ocr_result = ""
                for page in pages:
                    ocr_result += "\n".join(reader.readtext(np.array(page.convert("RGB")), detail=0)) + "\n"
            else:
                st.warning(f"{f.name} is not supported.")
                continue
            if any(k in f.name.lower() for k in ["pet", "ct", "xray", "scan"]):
                image_texts.append(ocr_result)
            else:
                lab_texts.append(ocr_result)
        except Exception as e:
            st.error(f"OCR failed for {f.name}: {e}")

# Show editable OCR text
if lab_texts:
    text_input = st.text_area("OCR extracted lab text (editable, English recommended)", value="\n".join(lab_texts), height=200)

input_source = text_input.strip() if text_input and text_input.strip() else None

# Lab parsing, health index, dietary deep dive functions remain the same

# Generate AI output
if st.button("Generate Summary & Recommendations"):
    st.session_state.clear()
    if not input_source and not lab_texts:
        st.error("Paste a lab excerpt or upload files first.")
    elif not client:
        st.error("OpenAI client not configured.")
    else:
        all_lab_text = input_source if input_source else "\n".join(lab_texts)
        health_index = compute_health_index_with_imaging(all_lab_text, image_texts)
        st.session_state['health_index'] = health_index
        st.subheader("Health Index")
        st.write(f"Combined Health Index (0-100): {health_index}")

        full_prompt = (
            "You are a clinical-support assistant. Respond only in English.\n"
            "Given the patient's report text below, produce exactly three labelled sections: Summary, Questions, Nutrition.\n"
            "Summary: 3-4 short sentences in plain, simple English.\n"
            "Questions: 3 practical questions to ask the doctor next visit.\n"
            "Nutrition: 3 dietitian-level, food-based nutrition recommendations (Malaysia Cancer Nutrition Guidelines), clear and detailed.\n"
            f"Patient report:\n{all_lab_text}"
        )

        with st.spinner("Generating AI output..."):
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user","content":full_prompt}],
                    max_tokens=700,
                    temperature=0.2
                )
                ai_text = resp.choices[0].message.content if 'choices' in resp else str(resp)

                summary = extract_section(ai_text, "Summary") or "No findings."
                questions = extract_section(ai_text, "Questions") or "No findings."
                nutrition = extract_section(ai_text, "Nutrition") or "No findings."

                st.session_state.update({'summary':summary, 'questions':questions, 'nutrition':nutrition, 'ai_raw':ai_text})

                st.subheader("Health Summary")
                st.write(summary)
                st.subheader("Suggested Questions for the Doctor")
                st.write(questions)
                st.subheader("Nutrition Recommendations")
                st.write(nutrition)

                labs = parse_lab_values(all_lab_text)
                st.json(labs)

                deep_text, flag = generate_dietary_deep_dive_en(labs)
                if flag:
                    st.subheader("Dietary Deep Dive (English, copyable)")
                    st.text_area("Dietary Deep Dive (English)", value=deep_text, height=320)
                    st.session_state['deep_en'] = deep_text

                with st.expander("Full AI output (raw)"):
                    st.code(ai_text)

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")

# Optional: send final report via email
st.subheader("Send report via email (optional)")
email_to = st.text_input("Recipient email (optional)")
send_email_btn = st.button("Send Report via Email")
if send_email_btn:
    if not email_to:
        st.error("Please provide a recipient email.")
    else:
        try:
            msg = EmailMessage()
            msg['Subject'] = 'Personalized Health Report'
            msg['From'] = st.secrets.get('EMAIL_SENDER')
            msg['To'] = email_to
            body = (
                f"Health Index: {st.session_state.get('health_index','N/A')}\n\n"
                f"Summary:\n{st.session_state.get('summary','No findings.')}\n\n"
                f"Questions:\n{st.session_state.get('questions','No findings.')}\n\n"
                f"Nutrition:\n{st.session_state.get('nutrition','No findings.')}\n\n"
                f"Dietary Deep Dive:\n{st.session_state.get('deep_en','No deep dive.')}")
            msg.set_content(body)
            with smtplib.SMTP_SSL(st.secrets.get('EMAIL_SMTP_SERVER'), st.secrets.get('EMAIL_SMTP_PORT')) as server:
                server.login(st.secrets.get('EMAIL_SENDER'), st.secrets.get('EMAIL_PASSWORD'))
                server.send_message(msg)
            st.success(f"Report sent to {email_to}")
        except Exception as e:
            st.error(f"Failed to send email: {e}")

# Optional: follow-up questions
st.subheader("Ask more questions to AI")
user_q = st.text_area("Enter follow-up questions here (optional)")
if st.button("Get AI Follow-up Answer"):
    if not user_q.strip():
        st.error("Please enter a question.")
    elif not client:
        st.error("OpenAI client not configured.")
    else:
        followup_prompt = (
            f"You are a clinical-support assistant. Respond in English. Patient previously received this report:\n{all_lab_text}\n"
            f"Now answer the follow-up question in detail, with clear explanation and dietitian-level nutrition guidance if relevant:\n{user_q}"
        )
        with st.spinner("Generating follow-up answer..."):
            try:
                resp2 = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user","content":followup_prompt}],
                    max_tokens=700,
                    temperature=0.2
                )
                followup_text = resp2.choices[0].message.content if 'choices' in resp2 else str(resp2)
                st.subheader("Follow-up Answer")
                st.write(followup_text)
            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")




























