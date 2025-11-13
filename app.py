import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import io
import openai
import requests
import json
import smtplib
from email.message import EmailMessage

# ------------------------- CONFIG -------------------------
st.set_page_config(page_title="AI Cancer Chatbot", layout="wide")

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Mailgun secrets
MAILGUN_API_KEY = st.secrets.get("MAILGUN_API_KEY")
MAILGUN_DOMAIN = st.secrets.get("MAILGUN_DOMAIN")
EMAIL_SENDER = f"postmaster@{MAILGUN_DOMAIN}" if MAILGUN_DOMAIN else None

# ------------------------- OCR -------------------------
reader = easyocr.Reader(['en'])

def ocr_extract(uploaded_files):
    texts = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        texts.append(" ".join(reader.readtext(np.array(img), detail=0)))
    return "\n".join(texts)

# ------------------------- AI INTERACTION -------------------------
def generate_ai_response(prompt, chat_history=None):
    if chat_history is None:
        chat_history = []
    messages = [{"role": "system", "content": "You are an expert oncology dietitian and medical assistant."}]
    for user_msg, ai_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": prompt})
    
    response = openai.ChatCompletion.create(
        model="gpt-5-mini",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# ------------------------- EMAIL -------------------------
def send_email(subject, body, to_email):
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        st.warning("Mailgun not configured. Email not sent.")
        return False
    try:
        url = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages"
        response = requests.post(
            url,
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": EMAIL_SENDER,
                "to": to_email,
                "subject": subject,
                "text": body
            }
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# ------------------------- UI -------------------------
st.title("AI-Driven Personalized Cancer Care Chatbot")

uploaded_files = st.file_uploader(
    "Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, dietary advice.",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("OCR Preview")
    for f in uploaded_files:
        st.image(f, caption=f.name, use_column_width=True)
    try:
        ocr_text = ocr_extract(uploaded_files)
        st.text_area("Extracted Text", ocr_text, height=200)
    except Exception as e:
        st.error(f"OCR extraction failed: {e}")

chat_history = st.session_state.get("chat_history", [])

prompt_generate = st.button("Generate Summary & Advice")

if prompt_generate and uploaded_files:
    with st.spinner("Generating health summary, doctor questions, and dietary advice..."):
        summary = generate_ai_response(f"Please generate a concise health summary, suggested questions for the doctor, and a structured dietitian-level dietary advice based on the following lab/report text:\n{ocr_text}")
        st.session_state["last_summary"] = summary
        chat_history.append((ocr_text, summary))
        st.session_state["chat_history"] = chat_history
        st.success("Done!")

# ------------------------- Chat/Refine Advice -------------------------
st.subheader("Chat & Refine Advice")
user_question = st.text_input("Ask follow-up / refine advice (for dietary or doctor questions)")

if st.button("Send Question"):
    if user_question:
        last_summary_text = st.session_state.get("last_summary", ocr_text if uploaded_files else "")
        with st.spinner("Processing your follow-up question..."):
            response = generate_ai_response(
                f"Refine the previous advice based on the following question: {user_question}\nPrevious summary/advice:\n{last_summary_text}",
                chat_history=chat_history
            )
            chat_history.append((user_question, response))
            st.session_state["chat_history"] = chat_history
            st.text_area("Chat History", "\n\n".join([f"User: {u}\nAI: {a}" for u,a in chat_history]), height=300)

# ------------------------- Send Email -------------------------
st.subheader("Send Final Report via Email (Optional)")
recipient_email = st.text_input("Recipient Email Address")
if st.button("Send Email"):
    if recipient_email:
        final_report = "\n\n".join([f"User: {u}\nAI: {a}" for u,a in chat_history])
        if send_email("AI Cancer Chatbot Report", final_report, recipient_email):
            st.success(f"Email sent successfully to {recipient_email}")
        else:
            st.error("Email failed to send.")
















































