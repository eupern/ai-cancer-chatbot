import streamlit as st
import numpy as np
import easyocr
from PIL import Image
import openai

# Load secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
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

# --- Generate summary & dietary advice ---
st.subheader("Generate summary & dietary advice")
col1, col2 = st.columns([3,1])
with col1:
    if st.button("Generate Summary & Dietary Advice"):
        if not st.session_state.ocr_text:
            st.warning("Please upload at least one document first.")
        else:
            prompt = (
                "Please generate a concise health summary (2-4 short paragraphs), a short list of suggested questions for the doctor (bullet points), "
                "and a structured, dietitian-level dietary advice with one-day sample menu. Use plain, actionable language. Base the answer only on the text below.\n\n"
                + st.session_state.ocr_text
            )
            with st.spinner("AI is generating summary and dietary advice..."):
                try:
                    response = openai.chat.completions.create(
                        model="gpt-5-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=800
                    )
                    ai_output = response.choices[0].message.content.strip()
                    st.session_state.conversation.append({"role": "assistant", "content": ai_output})
                    st.session_state.generated_summary = ai_output
                    st.success("Summary generated and added to chat.")
                except Exception as e:
                    st.error(f"AI generation failed: {e}")

with col2:
    if st.session_state.generated_summary:
        st.download_button("Download summary (txt)", st.session_state.generated_summary, file_name="summary.txt")

# --- Chat section (left: AI, right: User) ---
st.subheader("Chat with AI to refine dietary advice or doctor questions")

# Display chat history
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

# Input box at the bottom
chat_input = st.text_area("Type a follow-up or refinement and press Send:", "", height=100)
if st.button("Send Message"):
    if chat_input.strip() != "":
        st.session_state.conversation.append({"role": "user", "content": chat_input.strip()})
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.conversation]
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

# --- Export last assistant message and full conversation ---
st.subheader("Export / Share Conversation")
if st.session_state.conversation:
    last_ai = next((m["content"] for m in reversed(st.session_state.conversation) if m["role"]=="assistant"), "")
    st.text_area("Last AI message (copy & share):", last_ai, height=150)
    full_text = "\n\n".join([f"{m['role'].upper()}:\n{m['content']}" for m in st.session_state.conversation])
    st.download_button("Download full conversation (txt)", full_text, file_name="full_conversation.txt")




























































