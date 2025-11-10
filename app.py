# app.py
import streamlit as st
import os
import openai
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr

# Initialize EasyOCR reader (Simplified Chinese + English)
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")

st.title("🧠 AI-Driven Personalized Cancer Care Chatbot")
st.write(
    "Upload a medical report (JPG/PNG/PDF) or paste a test/result text. Click Generate to get: health summary, doctor questions, and nutrition advice."
)

# ===== OpenAI API key section =====
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# If API key is not found, allow temporary manual entry
if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Add OPENAI_API_KEY in Streamlit Secrets.")
    api_key_input = st.text_input(
        "Or paste your OpenAI API key for this session (not stored):",
        type="password"
    )
    if api_key_input:
        OPENAI_API_KEY = api_key_input

# Configure key
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# ===== Input area =====
st.subheader("1) Input report or summary")
uploaded_file = st.file_uploader("Upload a medical report (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])
text_input = st.text_area("Or paste a short lab summary here", height=160)

# ===== OCR extraction logic with EasyOCR =====
ocr_text = ""
if uploaded_file:
    try:
        # Handle images
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded report preview", use_column_width=True)

            # Convert image to array for OCR
            result = reader.readtext(np.array(image), detail=0)
            ocr_text = "\n".join(result)

        # Handle PDF files (convert to images first)
        elif uploaded_file.type == "application/pdf":
            pages = convert_from_bytes(uploaded_file.read())
            for page in pages:
                result = reader.readtext(np.array(page), detail=0)
                ocr_text += "\n".join(result) + "\n"

        else:
            st.warning("Unsupported file type for OCR.")
    except Exception as e:
        st.error(f"OCR processing failed: {e}")

# If OCR text exists, show and allow editing
if ocr_text.strip():
    text_input = st.text_area("OCR extracted text (editable)", value=ocr_text, height=200)

# Final text source (either OCR or manual input)
input_source = text_input.strip() if text_input and text_input.strip() else None

# ===== Button to call OpenAI =====
if st.button("Generate Summary & Recommendations"):
    if not input_source:
        st.error("Please provide report text or upload a file for OCR.")
    elif not OPENAI_API_KEY:
        st.error("OpenAI API key missing.")
    else:
        with st.spinner("Generating AI output..."):
            prompt = f"""
You are a helpful clinical-support assistant. Given the patient report below, produce:

1) A concise health summary in plain language (3–4 short sentences).
2) Three practical questions the patient/family should ask the doctor during the next appointment.
3) Three nutrition recommendations based on Malaysia Cancer Nutrition Guidelines (food-based and simple).

Patient report:
\"\"\"{input_source}\"\"\"

Output format:
Summary:
- ...
Questions:
- ...
Nutrition:
- ...
Keep the language simple and easily understood by elderly patients and family members.
"""

            try:
                # Use ChatCompletion API
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.2
                )

                # ===== Compatible response parsing for different OpenAI SDK versions =====
                ai_text = ""
                try:
                    # New OpenAI SDK format
                    ai_text = resp.choices[0].message.content
                except Exception:
                    try:
                        # Legacy dict-access format
                        ai_text = resp["choices"][0]["message"]["content"]
                    except Exception:
                        try:
                            ai_text = resp.choices[0].text
                        except Exception:
                            ai_text = str(resp)  # fallback

                # ===== Display section: Summary =====
                st.subheader("🧾 Health Summary")
                if "Summary:" in ai_text:
                    summary = ai_text.split("Summary:")[1].split("Questions:")[0].strip()
                    st.write(summary)
                else:
                    st.write(ai_text)

                # ===== Display section: Questions =====
                st.subheader("❓ Suggested Questions for the Doctor")
                if "Questions:" in ai_text:
                    questions = ai_text.split("Questions:")[1].split("Nutrition:")[0].strip()
                    st.write(questions)
                else:
                    st.write("Questions section not detected. See summary.")

                # ===== Display section: Nutrition =====
                st.subheader("🥗 Nutrition Recommendations")
                if "Nutrition:" in ai_text:
                    nutrition = ai_text.split("Nutrition:")[1].strip()
                    st.write(nutrition)
                else:
                    st.write("Nutrition section not clearly detected.")

                # ===== Always show raw output for debugging =====
                with st.expander("Full AI output (raw)"):
                    st.code(ai_text)

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")






