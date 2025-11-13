import streamlit as st
import numpy as np
import easyocr
from PIL import Image
import openai

# Load secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
# Mailgun removed per user request — email functionality replaced with copy/export options

openai.api_key = OPENAI_API_KEY

@st.cache_resource
def get_reader():
    return easyocr.Reader(['en'])

reader = get_reader()

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
            st.session_state.ocr_text = "

".join(combined_text)
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
                "and a structured, dietitian-level dietary advice with one-day sample menu. Use plain, actionable language. Base the answer only on the text below.

" + st.session_state.ocr_text
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
st.subheader("Chat — refine advice or ask follow-up questions
Type a follow-up or refinement and press Send:")

# Display conversation first (so input is at the bottom)
chat_area = st.container()
with chat_area:
    for msg in st.session_state.conversation:
        if msg['role'] == 'assistant':
            colL, colR = st.columns([0.7, 0.3])
            with colL:
                st.markdown("**AI**")
                st.info(msg['content'])
            with colR:
                st.write("")
        else:
            colL, colR = st.columns([0.3, 0.7])
            with colL:
                st.write("")
            with colR:
                st.markdown("**You**")
                st.write(msg['content'])

# --- Input area placed after conversation so it's always at the bottom ---
chat_input = st.text_area("", "", height=120, key="chat_input")
if st.button("Send Message"):
    if chat_input.strip() != "":
        # Append user message and clear input
        st.session_state.conversation.append({"role": "user", "content": chat_input.strip()})
        st.session_state.chat_input = ""
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

st.markdown("---")

# --- Export / copy options (replaced email) ---
st.subheader("Share / Export")
if st.session_state.conversation:
    # show last assistant message for easy copy
    last_assistant = ""
    for m in reversed(st.session_state.conversation):
        if m['role'] == 'assistant':
            last_assistant = m['content']
            break

    st.write("Last assistant message (you can copy & paste to share):")
    st.text_area("Copy-ready summary", value=last_assistant, height=200)

    # Download full conversation option
    final_report = []
    for m in st.session_state.conversation:
        final_report.append(f"{m['role'].upper()}:
{m['content']}")
    full_text = "

".join(final_report)
    st.download_button("Download full conversation (txt)", full_text, file_name="conversation.txt")

with st.expander("Helpful actions / notes", expanded=False):
    st.write("• Generate the assistant summary first, then use the copy box or download button to share with family or clinicians.")
    st.write("• The chat input is intentionally placed below the conversation so new messages are always added from the bottom.")
    st.write("• If OCR is poor for PDFs, try exporting pages as PNG/JPEG and re-uploading.")

# End of file


























































