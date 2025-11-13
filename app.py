import streamlit as st
import easyocr
import pandas as pd
import openai
import requests
from PIL import Image
import io

# ----------------- Streamlit Config -----------------
st.set_page_config(page_title="AI Cancer Chatbot", layout="wide")

# ----------------- Mailgun Settings -----------------
MAILGUN_API_KEY = st.secrets["MAILGUN_API_KEY"]
MAILGUN_DOMAIN = st.secrets["MAILGUN_DOMAIN"]
EMAIL_SENDER = f"postmaster@{MAILGUN_DOMAIN}"

# ----------------- OpenAI Key -----------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ----------------- Session State -----------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ----------------- Helper Functions -----------------
def send_mailgun_email(to_email, subject, text):
    return requests.post(
        f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
        auth=("api", MAILGUN_API_KEY),
        data={"from": EMAIL_SENDER,
              "to": [to_email],
              "subject": subject,
              "text": text})

# Function to generate AI response
def ai_generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a professional oncology nutrition and medical assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1200
    )
    return response.choices[0].message.content.strip()

# OCR Processing
def extract_text_from_image(uploaded_file):
    reader = easyocr.Reader(['en'])
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    result = reader.readtext(np.array(image), detail=0)
    return ' '.join(result)

# ----------------- Streamlit UI -----------------
st.title("AI-Driven Personalized Cancer Care Chatbot")

uploaded_files = st.file_uploader(
    label="Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, nutrition advice", 
    type=["jpg", "png", "pdf"], 
    accept_multiple_files=True)

user_question = st.text_area("Ask follow-up / refine dietary advice or doctor questions")

if st.button("Generate Summary & Advice"):
    all_text = []
    for f in uploaded_files:
        try:
            all_text.append(extract_text_from_image(f))
        except:
            all_text.append(f.read().decode("utf-8"))
    combined_text = '\n'.join(all_text)
    
    prompt = f"Medical report text: {combined_text}. Provide a professional health summary, dietitian-level dietary advice with 1-day menu, and suggested questions for the doctor."
    ai_response = ai_generate_response(prompt)
    st.session_state.chat_history.append(("User", "Initial Upload"))
    st.session_state.chat_history.append(("AI", ai_response))

if user_question:
    prompt_followup = f"Refine dietary advice or doctor questions based on user follow-up: {user_question}"
    ai_response_followup = ai_generate_response(prompt_followup)
    st.session_state.chat_history.append(("User", user_question))
    st.session_state.chat_history.append(("AI", ai_response_followup))

# Chat display
st.subheader("Chat History")
for role, msg in st.session_state.chat_history:
    if role == "User":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**AI:** {msg}")

# Email sending
st.subheader("Send Final Report via Email")
to_email = st.text_input("Recipient Email")
if st.button("Send Email") and to_email:
    full_report = '\n\n'.join([f"{role}: {msg}" for role, msg in st.session_state.chat_history])
    result = send_mailgun_email(to_email, "Your AI Cancer Chatbot Report", full_report)
    if result.status_code == 200:
        st.success("Email sent successfully!")
    else:
        st.error(f"Failed to send email: {result.text}")













































