import streamlit as st
import numpy as np
import easyocr
from PIL import Image
import openai
import requests

# Load secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MAILGUN_API_KEY = st.secrets.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = st.secrets.get("MAILGUN_DOMAIN", "")
EMAIL_SENDER = f"postmaster@{MAILGUN_DOMAIN}" if MAILGUN_DOMAIN else None

openai.api_key = OPENAI_API_KEY
reader = easyocr.Reader(['en'])

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "generated_summary" not in st.session_state:
    st.session_state.generated_summary = ""

st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="wide")
st.title("AI-Driven Personalized Cancer Care Chatbot")

# --- Upload documents and OCR ---
with st.expander("Upload medical documents (PDF or images)", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload medical documents (PDF or images):", type=['pdf','png','jpg','jpeg'], accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader("Uploaded Document Preview")
        combined_text = []
        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file)
                st.image(img, use_column_width=False, width=300)
                text = " ".join(reader.readtext(np.array(img), detail=0))
                if text:
                    combined_text.append(text)
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")

        if combined_text:
            st.session_state.ocr_text = "\n\n".join(combined_text)
            st.success("Document text extracted. You can generate a summary below.")

# --- Generate summary and structured dietary advice ---
st.subheader("Generate summary & dietary advice")
col1, col2 = st.columns([3,1])
with col1:
    if st.button("Generate Summary & Dietary Advice"):
        if not st.session_state.ocr_text:
            st.warning("Please upload at least one document first.")
        else:
            prompt = (
                "Please generate a concise health summary (2-4 short paragraphs), a short list of suggested questions for the doctor (bullet points), "
                "and a structured, dietitian-level dietary advice with one-day sample menu. Use plain, actionable language. Base the answer only on the text below.\n\n" + st.session_state.ocr_text
            )
            with st.spinner("AI is generating summary and dietary advice..."):
                try:
                    response = openai.chat.completions.create(
                        model="gpt-5-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=800
                    )
                    ai_output = response.choices[0].message.content.strip()
                    # Store assistant output and keep conversation flow
                    st.session_state.conversation.append({"role": "assistant", "content": ai_output})
                    st.session_state.generated_summary = ai_output
                    st.success("Summary generated and added to chat.")
                except Exception as e:
                    st.error(f"AI generation failed: {e}")

with col2:
    st.write("")
    if st.session_state.generated_summary:
        if st.download_button("Download summary (txt)", st.session_state.generated_summary, file_name="summary.txt"):
            pass

# --- Chat section (left: AI, right: User style) ---
st.subheader("Chat with AI to refine dietary advice or doctor questions")
chat_input = st.text_area("Your message:", "", height=100)

if st.button("Send Message"):
    if chat_input.strip() != "":
        # Append user message
        st.session_state.conversation.append({"role": "user", "content": chat_input.strip()})
        # Build messages for the model from conversation
        messages = []
        for msg in st.session_state.conversation:
            # Map to model roles
            if msg['role'] == 'assistant':
                messages.append({"role": "assistant", "content": msg['content']})
            else:
                messages.append({"role": "user", "content": msg['content']})
        with st.spinner("AI is thinking..."):
            try:
                response = openai.chat.completions.create(
                    model="gpt-5-mini",
                    messages=messages,
                    max_tokens=500
                )
                ai_reply = response.choices[0].message.content.strip()
                st.session_state.conversation.append({"role": "assistant", "content": ai_reply})
            except Exception as e:
                st.error(f"AI follow-up generation failed: {e}")

# Custom visual chat display: AI messages left, User messages right
st.markdown("---")
chat_area = st.container()
with chat_area:
    for msg in st.session_state.conversation:
        if msg['role'] == 'assistant':
            colL, colR = st.columns([0.7, 0.3])
            with colL:
                st.markdown(f"**AI**")
                st.info(msg['content'])
            with colR:
                st.write("")
        else:
            colL, colR = st.columns([0.3, 0.7])
            with colL:
                st.write("")
            with colR:
                st.markdown(f"**You**")
                st.write(msg['content'])

# --- Mail flow: send only assistant-generated report or full conversation ---
st.subheader("Send Report via Email (Optional)")
email_to = st.text_input("Recipient Email")
send_mode = st.radio("Report content", ["Assistant-only summary (recommended)", "Full conversation"], index=0)

if st.button("Send Email"):
    if not email_to:
        st.warning("Please provide a recipient email address.")
    elif not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        st.error("Mailgun settings missing in Streamlit secrets. Please add MAILGUN_API_KEY and MAILGUN_DOMAIN.")
    else:
        try:
            if send_mode == "Assistant-only summary (recommended)":
                if not st.session_state.generated_summary:
                    st.warning("No assistant summary available — generate one first.")
                else:
                    body_text = st.session_state.generated_summary
                    subject = "Personalized Health Summary and Dietary Advice"
            else:
                # Full conversation
                final_report = []
                for m in st.session_state.conversation:
                    final_report.append(f"{m['role'].upper()}:\n{m['content']}")
                body_text = "\n\n".join(final_report)
                subject = "Full AI-Patient Conversation Export"

            if body_text:
                resp = requests.post(
                    f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                    auth=("api", MAILGUN_API_KEY),
                    data={
                        "from": EMAIL_SENDER,
                        "to": [email_to],
                        "subject": subject,
                        "text": body_text
                    }
                )
                if resp.status_code in (200, 202):
                    st.success(f"Email successfully sent to {email_to}")
                else:
                    st.error(f"Failed to send email: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f"Email sending error: {e}")

# --- Helpful notes and quick actions ---
with st.expander("Helpful actions / notes", expanded=False):
    st.write("• Generate the assistant summary first, then use the Assistant-only email mode to send a concise report to clinicians or family.")
    st.write("• Use the chat box to refine the AI's suggested doctor questions or dietary plan. Each Send will append to the conversation and prompt the AI for a follow-up reply.")
    st.write("• If OCR is poor for PDFs, try exporting pages as PNG/JPEG and re-uploading.")

# End of file

























































