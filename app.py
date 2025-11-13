import streamlit as st
import os
import openai
import easyocr
import numpy as np
from PIL import Image
import requests

# ------------------ SECRETS ------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]
MAILGUN_API_KEY = st.secrets["MAILGUN_API_KEY"]
MAILGUN_DOMAIN = st.secrets["MAILGUN_DOMAIN"]
EMAIL_SENDER = f"postmaster@{MAILGUN_DOMAIN}"

# ------------------ SESSION STATE ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ OCR FUNCTION ------------------
reader = easyocr.Reader(['en'])
def ocr_extract(uploaded_files):
    texts = []
    for file in uploaded_files:
        img = Image.open(file)
        texts.append(" ".join(reader.readtext(np.array(img), detail=0)))
    return "\n".join(texts)

# ------------------ AI FUNCTION ------------------
def generate_ai_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-5-mini",
        messages=messages,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# ------------------ UI ------------------
st.title("AI-Driven Personalized Cancer Care Chatbot")

uploaded_files = st.file_uploader(
    "Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, dietary advice.",
    type=["jpg","png","pdf"],
    accept_multiple_files=True
)

user_input = st.text_area("Or paste lab/report excerpt here:")

# ------------------ GENERATE BUTTON ------------------
if st.button("Generate Summary & Advice"):
    ocr_text = ocr_extract(uploaded_files) if uploaded_files else user_input
    if not ocr_text.strip():
        st.warning("Please upload or paste lab/report data first.")
    else:
        prompt = f"""Please generate a concise health summary, suggested questions for the doctor, 
        and a structured dietitian-level dietary advice based on the following lab/report text:\n{ocr_text}"""
        
        ai_response = generate_ai_response(prompt)
        st.session_state.chat_history.append({"role":"AI","content": ai_response})

# ------------------ FOLLOW-UP ------------------
followup = st.text_input("Ask follow-up / refine dietary advice or doctor questions:")
if st.button("Send Follow-up"):
    if followup.strip():
        followup_prompt = f"""Refine the previous AI report based on this follow-up question:\n{followup}\n
        Previous report:\n{st.session_state.chat_history[-1]['content']}"""
        ai_followup = generate_ai_response(followup_prompt)
        st.session_state.chat_history.append({"role":"User","content": followup})
        st.session_state.chat_history.append({"role":"AI","content": ai_followup})

# ------------------ CHAT / A4 REPORT ------------------
st.subheader("AI Health Report (A4 Layout Preview)")
if st.session_state.chat_history:
    report_container = st.container()
    with report_container:
        st.markdown(
            """
            <div style="
                width: 800px; 
                min-height: 1123px;  
                padding: 40px;
                border: 1px solid #ccc;
                margin: auto;
                font-family: Arial, sans-serif;
                line-height: 1.5;
                background-color: #fff;
            ">
            """, unsafe_allow_html=True
        )
        for chat in st.session_state.chat_history:
            if chat["role"] == "AI":
                st.markdown(f"<p><b>AI:</b> {chat['content']}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p><b>You:</b> {chat['content']}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ EMAIL ------------------
st.subheader("Send Report via Email (Optional)")
recipient_email = st.text_input("Recipient Email:")
if st.button("Send Email"):
    if recipient_email.strip() and st.session_state.chat_history:
        try:
            import requests
            data = {
                "from": EMAIL_SENDER,
                "to": recipient_email,
                "subject": "AI Health Report",
                "text": "\n\n".join([c['content'] for c in st.session_state.chat_history if c['role']=='AI'])
            }
            response = requests.post(
                f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                auth=("api", MAILGUN_API_KEY),
                data=data
            )
            if response.status_code == 200:
                st.success("Email sent successfully!")
            else:
                st.error(f"Failed to send email: {response.text}")
        except Exception as e:
            st.error(f"Failed to send email: {e}")
    else:
        st.warning("Provide recipient email and generate report first.")

















































