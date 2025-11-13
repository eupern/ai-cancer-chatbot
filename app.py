import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import openai
import requests

# -----------------------------
# Streamlit secrets
# -----------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]
MAILGUN_API_KEY = st.secrets.get("MAILGUN_API_KEY")
MAILGUN_DOMAIN = st.secrets.get("MAILGUN_DOMAIN")
EMAIL_SENDER = f"postmaster@{MAILGUN_DOMAIN}"

# -----------------------------
# Constants
# -----------------------------
MAX_PROMPT_CHARS = 3000  # Limit for OCR text to avoid OpenAI BadRequest

# -----------------------------
# OCR function
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

def ocr_extract(uploaded_files):
    texts = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert('RGB')
        text = " ".join(reader.readtext(np.array(img), detail=0))
        texts.append(text)
    combined_text = " ".join(texts)
    # Truncate if too long
    return combined_text[:MAX_PROMPT_CHARS]

# -----------------------------
# AI Chat Response
# -----------------------------
def generate_ai_response(prompt_text, conversation=None):
    if conversation is None:
        conversation = []

    conversation.append({"role": "user", "content": prompt_text})
    try:
        response = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=conversation,
            temperature=0.7
        )
        ai_message = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": ai_message})
        return ai_message, conversation
    except Exception as e:
        st.error(f"AI generation failed: {e}")
        return "", conversation

# -----------------------------
# Mailgun Email Sending
# -----------------------------
def send_email_via_mailgun(to_email, subject, text):
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN):
        st.warning("Mailgun secrets missing. Cannot send email.")
        return False
    try:
        response = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": EMAIL_SENDER,
                "to": [to_email],
                "subject": subject,
                "text": text
            }
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(layout="wide", page_title="AI Cancer Health Chatbot")

st.title("AI-Driven Personalized Health Summary & Dietary Advice")
st.markdown(
    "Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. "
    "Click Generate to get an English health summary, suggested questions for the doctor, and dietitian-level dietary advice."
)

# Upload section
uploaded_files = st.file_uploader(
    "Upload Medical Reports", type=["jpg", "png", "pdf"], accept_multiple_files=True
)

ocr_text = ""
if uploaded_files:
    with st.spinner("Running OCR on uploaded files..."):
        ocr_text = ocr_extract(uploaded_files)
    st.subheader("Preview of OCR Extracted Text")
    st.text_area("OCR Preview", ocr_text, height=300)

# Chat flow
if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.subheader("AI Chat")
user_prompt = st.text_area("Ask follow-up or refine dietary advice / doctor questions:", "")

if st.button("Generate Summary & Advice"):
    if ocr_text.strip() == "" and user_prompt.strip() == "":
        st.warning("Please upload report or type a question.")
    else:
        prompt = f"Please generate a concise health summary, suggested questions for the doctor, and a structured, dietitian-level dietary advice based on the following lab/report text:\n{ocr_text}"
        ai_message, st.session_state.conversation = generate_ai_response(prompt, st.session_state.conversation)
        st.success("AI Response generated and added to chat.")

if user_prompt.strip() != "" and st.button("Send Follow-up / Refine Advice"):
    ai_message, st.session_state.conversation = generate_ai_response(user_prompt, st.session_state.conversation)
    st.success("Follow-up response generated and added to chat.")

# Display chat conversation
for msg in st.session_state.conversation:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(f"**You:** {content}")
    else:
        st.markdown(f"**AI:** {content}")

# Optional email sending
st.subheader("Send Final Report via Email (Optional)")
recipient_email = st.text_input("Recipient Email")
if st.button("Send Email") and recipient_email:
    full_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation])
    success = send_email_via_mailgun(recipient_email, "Your Health Summary & Dietary Advice", full_text)
    if success:
        st.success("Email sent successfully!")
    else:
        st.error("Failed to send email.")




















































