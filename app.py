import streamlit as st
import numpy as np
import easyocr
from PIL import Image
import openai

# Load secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
reader = easyocr.Reader(['en'])

# Chat session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

st.title("AI-Driven Personalized Cancer Care Chatbot")

# Upload medical documents
uploaded_files = st.file_uploader(
    "Upload medical documents (PDF or images):", type=['pdf','png','jpg','jpeg'], accept_multiple_files=True
)

if uploaded_files:
    st.subheader("Uploaded Document Preview")
    combined_text = []
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file)
            st.image(img, use_column_width=True)
            text = " ".join(reader.readtext(np.array(img), detail=0))
            if text:
                combined_text.append(text)
        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")
    if combined_text:
        st.session_state.ocr_text = "\n\n".join(combined_text)
        st.success("Document text extracted. You can generate a summary below.")

# Generate summary & advice
if st.button("Generate Summary & Dietary Advice"):
    if not st.session_state.ocr_text:
        st.warning("Please upload at least one document first.")
    else:
        prompt = (
            f"Please generate a concise health summary, suggested questions for the doctor, "
            f"and a structured, dietitian-level dietary advice (with 1-day sample menu) "
            f"based on the following lab/report text:\n\n{st.session_state.ocr_text}"
        )
        with st.spinner("AI is generating summary and dietary advice..."):
            try:
                response = openai.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                ai_output = response.choices[0].message.content
                # Add as first AI message
                st.session_state.conversation.append({"role": "assistant", "content": ai_output})
            except Exception as e:
                st.error(f"AI generation failed: {e}")

# Chat input at bottom
st.subheader("Chat with AI to refine dietary advice or doctor questions")
chat_input = st.text_area("Type a follow-up or refinement and press Send:", "", height=100)
if st.button("Send Message"):
    if chat_input.strip():
        st.session_state.conversation.append({"role": "user", "content": chat_input})
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

# Display chat history: AI left, user right
st.markdown("---")
for msg in st.session_state.conversation:
    if msg["role"] == "assistant":
        colL, colR = st.columns([0.7, 0.3])
        with colL:
            st.markdown("**AI:**")
            st.info(msg["content"])
        with colR:
            st.write("")
    else:
        colL, colR = st.columns([0.3, 0.7])
        with colL:
            st.write("")
        with colR:
            st.markdown("**You:**")
            st.write(msg["content"])

# Download full conversation
if st.session_state.conversation:
    full_conversation = "\n\n".join([f"{m['role'].upper()}:\n{m['content']}" for m in st.session_state.conversation])
    st.download_button(
        "Download Full Conversation (txt)",
        data=full_conversation,
        file_name="conversation.txt",
        mime="text/plain"
    )
































































