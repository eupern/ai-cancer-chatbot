# app.py
import streamlit as st
import os
import openai
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract

st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")

st.title("🧠 AI-Driven Personalized Cancer Care Chatbot")
st.write(
    "Upload a medical report (JPG/PNG/PDF) or paste a short test/result. Click Generate to get health summary, doctor questions, and nutrition advice."
)

# OpenAI API key
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Add OPENAI_API_KEY in Streamlit Secrets.")
    api_key_input = st.text_input(
        "Or paste your OpenAI API key for this session (will not be saved to GitHub):",
        type="password"
    )
    if api_key_input:
        OPENAI_API_KEY = api_key_input

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Input area
st.subheader("1) Input report or summary")
uploaded_file = st.file_uploader("Upload a medical report (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])
text_input = st.text_area("Or paste a short report / lab excerpt here", height=160)

# OCR extraction
ocr_text = ""
if uploaded_file:
    try:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded report (preview)", use_column_width=True)
            ocr_text = pytesseract.image_to_string(image, lang='eng+chi_sim')
        elif uploaded_file.type == "application/pdf":
            pages = convert_from_bytes(uploaded_file.read())
            for page in pages:
                ocr_text += pytesseract.image_to_string(page, lang='eng+chi_sim') + "\n"
        else:
            st.warning("Unsupported file type for OCR.")
    except Exception as e:
        st.error(f"OCR processing failed: {e}")

# 如果 OCR 有结果，自动填入 text_input
if ocr_text.strip():
    text_input = ocr_text
    text_input = st.text_area("OCR extracted text (editable)", value=text_input, height=160)

input_source = text_input.strip() if text_input.strip() else None

# Action button
if st.button("Generate Summary & Recommendations"):
    if not input_source:
        st.error("Please provide a report text or upload an OCR-compatible file.")
    elif not OPENAI_API_KEY:
        st.error("OpenAI API key not configured.")
    else:
        with st.spinner("Generating..."):
            prompt = f"""
You are a helpful clinical-support assistant. Given the following patient's lab/report text, produce:
1) A concise health summary in plain language (3-4 short sentences).
2) A short list (3) of relevant, practical questions the patient/family should ask the doctor next visit.
3) Personalized nutritional recommendations based on Malaysia Cancer Nutrition Guidelines — concise, specific food or habit suggestions (3 bullets).

Patient report / data:
\"\"\"{input_source}\"\"\"

Please output clearly labeled sections, for example:
Summary:
- ...
Questions:
- ...
Nutrition:
- ...
Keep language simple and actionable for elderly patients and family.
"""
            try:
                resp = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.2
                )
                ai_text = resp["choices"][0]["message"]["content"]

                st.subheader("🧾 Health Summary")
                if "Summary:" in ai_text:
                    summary = ai_text.split("Summary:")[1].split("Questions:")[0].strip()
                    st.write(summary)
                else:
                    st.write(ai_text)

                st.subheader("❓ Suggested Questions for the Doctor")
                if "Questions:" in ai_text:
                    questions = ai_text.split("Questions:")[1].split("Nutrition:")[0].strip()
                    st.write(questions)
                else:
                    st.write("See summary above.")

                st.subheader("🥗 Nutrition Recommendations")
                if "Nutrition:" in ai_text:
                    nutrition = ai_text.split("Nutrition:")[1].strip()
                    st.write(nutrition)
                else:
                    st.write("No nutrition section detected; see summary above.")

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")


