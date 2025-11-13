import streamlit as st
import openai
import easyocr
import numpy as np
from PIL import Image
import io
import smtplib
from email.message import EmailMessage

# ---- Secrets ----
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MAILGUN_API_KEY = st.secrets["MAILGUN_API_KEY"]
MAILGUN_DOMAIN = st.secrets["MAILGUN_DOMAIN"]
EMAIL_SENDER = st.secrets["EMAIL_SENDER"]

openai.api_key = OPENAI_API_KEY

# ---- OCR ----
reader = easyocr.Reader(['en'], gpu=False)

def ocr_extract(files):
    texts = []
    for uploaded_file in files:
        img = Image.open(uploaded_file)
        texts.append(" ".join(reader.readtext(np.array(img), detail=0)))
    return "\n".join(texts)

# ---- AI Response ----
def generate_ai_response(prompt, conversation=None):
    messages = []
    if conversation:
        messages = conversation
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model="gpt-5-mini",
        messages=messages,
        temperature=0.7
    )
    content = response['choices'][0]['message']['content']
    messages.append({"role": "assistant", "content": content})
    return content, messages

# ---- Email Sending via Mailgun SMTP ----
def send_email(to_email, subject, body):
    msg = EmailMessage()
    msg['From'] = EMAIL_SENDER
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.set_content(body)
    
    smtp_host = "smtp.mailgun.org"
    smtp_port = 587
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(f"postmaster@{MAILGUN_DOMAIN}", MAILGUN_API_KEY)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# ---- Streamlit Layout ----
st.set_page_config(layout="wide")
st.title("AI-Driven Personalized Cancer Care Chatbot")

uploaded_files = st.file_uploader(
    "Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, dietary advice",
    type=['png', 'jpg', 'jpeg', 'pdf'], accept_multiple_files=True
)

conversation = st.session_state.get("conversation", [])

if uploaded_files:
    with st.container():
        st.subheader("Uploaded Document Preview")
        for f in uploaded_files:
            if f.type.startswith("image/"):
                st.image(f, use_column_width=True)
            else:
                st.text(f.name)
                
    if st.button("Generate Summary & Dietary Advice"):
        ocr_text = ocr_extract(uploaded_files)
        prompt = f"Please generate a concise health summary, suggested questions for the doctor, and a structured dietitian-level dietary advice based on the following lab/report text:\n{ocr_text}"
        ai_response, conversation = generate_ai_response(prompt, conversation)
        st.session_state.conversation = conversation
        st.subheader("AI Chat Flow")
        for msg in conversation:
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**AI:** {msg['content']}")
        
st.subheader("Ask follow-up / Refine Advice")
followup_input = st.text_area("Ask follow-up question about dietary advice or doctor questions:", "")
if st.button("Send Follow-up"):
    if followup_input:
        ai_response, conversation = generate_ai_response(followup_input, conversation)
        st.session_state.conversation = conversation
        st.subheader("Updated Chat Flow")
        for msg in conversation:
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**AI:** {msg['content']}")

st.subheader("Email Summary (Optional)")
to_email = st.text_input("Recipient email")
if st.button("Send Email"):
    if to_email and conversation:
        body = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in conversation])
        send_email(to_email, "Your AI Health Summary & Dietary Advice", body)


















































