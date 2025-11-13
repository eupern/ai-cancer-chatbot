import streamlit as st
import easyocr
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import os
from datetime import datetime
import json

st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="wide")

# ---------- Streamlit Secrets ----------
MAILGUN_API_KEY = st.secrets.get("MAILGUN_API_KEY")
MAILGUN_DOMAIN = st.secrets.get("MAILGUN_DOMAIN")
EMAIL_SENDER = f"postmaster@{MAILGUN_DOMAIN}" if MAILGUN_DOMAIN else None

# ---------- Chat session ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- OCR function ----------
def ocr_extract(files):
    reader = easyocr.Reader(['en'])
    texts = []
    for file in files:
        img = Image.open(file)
        texts.append(" ".join(reader.readtext(np.array(img), detail=0)))
    return "\n".join(texts)

# ---------- AI interaction ----------
def ai_generate(prompt):
    import openai
    openai.api_key = st.secrets.get("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-5-mini",
        messages=[{"role":"system","content":"You are a professional medical assistant."},
                  {"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=800
    )
    return response.choices[0].message.content

# ---------- Mailgun send ----------
def send_email_mailgun(to_email, subject, body):
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        st.error("Mailgun secrets missing.")
        return
    return requests.post(
        f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
        auth=("api", MAILGUN_API_KEY),
        data={"from": EMAIL_SENDER,
              "to": [to_email],
              "subject": subject,
              "text": body}
    )

# ---------- Layout ----------
st.title("AI-Driven Personalized Cancer Care Chatbot")

uploaded_files = st.file_uploader(
    "Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, dietary advice.",
    type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True
)

user_text_input = st.text_area("Or ask follow-up / refine advice:")

if st.button("Generate Summary & Advice") or user_text_input:
    if uploaded_files:
        ocr_text = ocr_extract(uploaded_files)
    else:
        ocr_text = ""

    user_prompt = f"""
    Extract health summary, suggested doctor questions, and detailed dietitian-level dietary advice from the following medical text. 
    Include a 1-day practical menu at the end. Be structured. Always provide dietary advice, regardless of lab flags.
    {ocr_text}
    """

    # Append follow-up question if any
    if user_text_input:
        user_prompt += f"\n\nUser follow-up: {user_text_input}"

    ai_reply = ai_generate(user_prompt)

    # Store in chat
    st.session_state.chat_history.append({"user": user_text_input or "Generate Summary", "ai": ai_reply})

# ---------- Chat display ----------
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**AI:** {chat['ai']}")
    st.markdown("---")

# ---------- Export / Email ----------
email_to = st.text_input("Optional: send final report to email")
if st.button("Send Report"):
    final_report = "\n\n".join([f"You: {c['user']}\nAI: {c['ai']}" for c in st.session_state.chat_history])
    if email_to:
        res = send_email_mailgun(email_to, "Your AI Cancer Care Report", final_report)
        if res and res.status_code == 200:
            st.success("Email sent successfully.")
        else:
            st.error("Failed to send email. Check Mailgun domain/API key.")














































