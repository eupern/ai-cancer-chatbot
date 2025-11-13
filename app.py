import streamlit as st
import numpy as np
from PIL import Image
import easyocr
import openai
import requests
import json

# === Initialize OpenAI ===
openai.api_key = st.secrets["OPENAI_API_KEY"]

# === Mailgun setup ===
MAILGUN_DOMAIN = st.secrets["MAILGUN_DOMAIN"]
MAILGUN_API_KEY = st.secrets["MAILGUN_API_KEY"]
EMAIL_SENDER = f"postmaster@{MAILGUN_DOMAIN}"

# === OCR reader ===
reader = easyocr.Reader(['en'])

# === Session state for chat ===
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# === OCR extraction function ===
def ocr_extract(files):
    texts = []
    for uploaded_file in files:
        img = Image.open(uploaded_file)
        text = " ".join(reader.readtext(np.array(img), detail=0))
        texts.append(text)
    return "\n".join(texts)

# === AI response function ===
def generate_ai_response(prompt):
    messages = [{"role": "system", "content": "You are a helpful cancer-care health assistant."}]
    # Include previous conversation
    for c in st.session_state.conversation:
        messages.append({"role": c["role"], "content": c["content"]})
    messages.append({"role": "user", "content": prompt})

    response = openai.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        temperature=0.7
    )
    answer = response.choices[0].message.content.strip()
    st.session_state.conversation.append({"role": "assistant", "content": answer})
    return answer

# === Streamlit UI ===
st.title("AI-Driven Personalized Cancer Care Chatbot")
st.markdown(
    "Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. "
    "Click Generate to get an English health summary, suggested questions, and dietitian-level dietary advice."
)

# === File uploader ===
uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["png","jpg","jpeg","pdf"])

if uploaded_files:
    ocr_text = ocr_extract(uploaded_files)
    st.subheader("OCR Preview of Uploaded Files")
    st.text_area("Extracted Text", ocr_text, height=250)

# === User text input for manual lab/report text ===
manual_input = st.text_area("Or paste short lab/test excerpt here:")

# === Generate button ===
if st.button("Generate Summary & Advice"):
    prompt_text = manual_input if manual_input.strip() else ocr_text
    if prompt_text.strip() == "":
        st.warning("Please upload a file or enter text first.")
    else:
        summary = generate_ai_response(
            f"Please generate a concise health summary, suggested questions for the doctor, "
            f"and a structured, dietitian-level dietary advice based on the following lab/report text:\n{prompt_text}"
        )
        st.subheader("Chat-style Summary & Advice")
        for c in st.session_state.conversation:
            if c["role"] == "assistant":
                st.markdown(f"**AI:** {c['content']}")
            else:
                st.markdown(f"**You:** {c['content']}")

# === Follow-up question input ===
followup = st.text_input("Ask follow-up / refine dietary advice or doctor questions:")
if st.button("Send Follow-up"):
    if followup.strip():
        answer = generate_ai_response(followup)
        st.subheader("Chat Updated")
        for c in st.session_state.conversation:
            if c["role"] == "assistant":
                st.markdown(f"**AI:** {c['content']}")
            else:
                st.markdown(f"**You:** {c['content']}")

# === Email sending ===
st.subheader("Optional: Send Final Report via Email")
email_to = st.text_input("Recipient Email")
if st.button("Send Email"):
    if email_to.strip() == "":
        st.warning("Enter recipient email.")
    else:
        if len(st.session_state.conversation) == 0:
            st.warning("No chat content to send.")
        else:
            final_content = "\n\n".join([f"You: {c['content']}" if c["role"]=="user" else f"AI: {c['content']}" for c in st.session_state.conversation])
            try:
                resp = requests.post(
                    f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                    auth=("api", MAILGUN_API_KEY),
                    data={"from": EMAIL_SENDER,
                          "to": [email_to],
                          "subject": "AI-Generated Health Summary & Dietary Advice",
                          "text": final_content})
                if resp.status_code == 200:
                    st.success("Email sent successfully!")
                else:
                    st.error(f"Failed to send email: {resp.text}")
            except Exception as e:
                st.error(f"Failed to send email: {e}")



















































