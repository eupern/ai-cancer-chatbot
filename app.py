import streamlit as st
import numpy as np
import easyocr
from PIL import Image
import openai
import requests

# Load secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MAILGUN_API_KEY = st.secrets["MAILGUN_API_KEY"]
MAILGUN_DOMAIN = st.secrets["MAILGUN_DOMAIN"]
EMAIL_SENDER = f"postmaster@{MAILGUN_DOMAIN}"

openai.api_key = OPENAI_API_KEY
reader = easyocr.Reader(['en'])

# Chat session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.title("AI-Driven Personalized Cancer Care Chatbot")

# Upload medical documents
uploaded_files = st.file_uploader(
    "Upload medical documents (PDF or images):", type=['pdf','png','jpg','jpeg'], accept_multiple_files=True
)

ocr_text = ""
if uploaded_files:
    st.subheader("Uploaded Document Preview")
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file)
            st.image(img, use_column_width=True)
            ocr_text += " ".join(reader.readtext(np.array(img), detail=0)) + "\n"
        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")

# Generate summary & advice
if st.button("Generate Summary & Dietary Advice"):
    if not ocr_text:
        st.warning("Please upload at least one document first.")
    else:
        prompt = (
            f"Please generate a concise health summary, suggested questions for the doctor, "
            f"and a structured, dietitian-level dietary advice (with 1-day sample menu) based on the following lab/report text:\n\n{ocr_text}"
        )
        with st.spinner("AI is generating summary and dietary advice..."):
            try:
                response = openai.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                ai_output = response.choices[0].message.content
                st.session_state.conversation.append({"role": "assistant", "content": ai_output})
            except Exception as e:
                st.error(f"AI generation failed: {e}")

# Show conversation/chat
st.subheader("Chat with AI to refine dietary advice or doctor questions")
chat_input = st.text_area("Your message:", "")
if st.button("Send Message"):
    if chat_input.strip() != "":
        st.session_state.conversation.append({"role": "user", "content": chat_input})
        # Generate AI response
        with st.spinner("AI is thinking..."):
            try:
                response = openai.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.conversation]
                )
                ai_reply = response.choices[0].message.content
                st.session_state.conversation.append({"role": "assistant", "content": ai_reply})
            except Exception as e:
                st.error(f"AI follow-up generation failed: {e}")

# Display chat history
for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AI:** {msg['content']}")

# Optional: send final report via Mailgun
st.subheader("Send Report via Email (Optional)")
email_to = st.text_input("Recipient Email")
if st.button("Send Email"):
    if not email_to:
        st.warning("Please provide a recipient email address.")
    else:
        try:
            final_report = "\n\n".join([f"{m['role'].upper()}:\n{m['content']}" for m in st.session_state.conversation])
            response = requests.post(
                f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                auth=("api", MAILGUN_API_KEY),
                data={"from": EMAIL_SENDER,
                      "to": [email_to],
                      "subject": "Your Personalized Health Report",
                      "text": final_report}
            )
            if response.status_code == 200:
                st.success(f"Email successfully sent to {email_to}")
            else:
                st.error(f"Failed to send email: {response.text}")
        except Exception as e:
            st.error(f"Email sending error: {e}")


































































