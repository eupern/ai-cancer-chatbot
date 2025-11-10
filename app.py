# app.py
import streamlit as st
import os
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr

# Initialize EasyOCR reader (Simplified Chinese + English)
# GPU disabled for Streamlit Cloud compatibility
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# Streamlit page config
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")

st.title("🧠 AI-Driven Personalized Cancer Care Chatbot")
st.write(
    "Upload a medical report (JPG/PNG/PDF) or paste a short test/result. Click Generate to get a health summary, suggested doctor questions, and nutrition advice."
)

# ===== OpenAI API key and client setup =====
OPENAI_API_KEY = None
try:
    # Reads secret from Streamlit Secrets (recommended on Streamlit Cloud)
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    # fallback: environment variable (useful for local testing)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Allow manual input for quick testing (not stored in GitHub)
if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found in Streamlit Secrets. Add OPENAI_API_KEY in Secrets or paste a key below for a quick test.")
    api_key_input = st.text_input("Paste your OpenAI API key for this session (not saved):", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

client = None
if OPENAI_API_KEY:
    # Create the OpenAI client for openai>=1.0.0
    client = OpenAI(api_key=OPENAI_API_KEY)

# ===== Input UI =====
st.subheader("1) Input report or summary")
uploaded_file = st.file_uploader("Upload a medical report (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])
text_input = st.text_area("Or paste a short lab summary / excerpt here", height=160)

# ===== OCR extraction (EasyOCR) =====
ocr_text = ""
if uploaded_file:
    try:
        if uploaded_file.type.startswith("image"):
            # Show preview and run OCR on the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded report (preview)", use_column_width=True)
            result = reader.readtext(np.array(image), detail=0)
            ocr_text = "\n".join(result)
        elif uploaded_file.type == "application/pdf":
            # Convert PDF pages to images and run OCR page by page
            pages = convert_from_bytes(uploaded_file.read())
            for page in pages:
                page_arr = np.array(page)
                result = reader.readtext(page_arr, detail=0)
                ocr_text += "\n".join(result) + "\n"
        else:
            st.warning("Uploaded file type is not supported for OCR.")
    except Exception as e:
        # Show OCR error details so you can debug (e.g., missing poppler)
        st.error(f"OCR processing failed: {e}")

# If OCR found text, present it in an editable text area
if ocr_text.strip():
    text_input = st.text_area("OCR extracted text (editable)", value=ocr_text, height=200)

# Decide which text to send to the model
input_source = text_input.strip() if text_input and text_input.strip() else None

# ===== Button: call OpenAI model =====
if st.button("Generate Summary & Recommendations"):
    if not input_source:
        st.error("Please paste a short report excerpt or upload an OCR-compatible file first.")
    elif not client:
        st.error("OpenAI client not configured. Please set OPENAI_API_KEY in Streamlit Secrets or paste it above.")
    else:
        with st.spinner("Generating AI output..."):
            # Prompt instructs the model to produce labelled sections
            prompt = f"""
You are a clinical-support assistant. Given the patient's report text below, produce:
1) A concise health summary in plain language (3-4 short sentences).
2) Three practical questions the patient/family should ask the doctor at the next visit.
3) Three personalized, food-based nutrition recommendations based on Malaysia Cancer Nutrition Guidelines (simple and actionable).

Patient report:
\"\"\"{input_source}\"\"\"

Output format (use these labels):
Summary:
- ...
Questions:
- ...
Nutrition:
- ...
Keep the language simple and suitable for elderly patients and family members.
"""

            try:
                # Use the new OpenAI client API (compatible with openai>=1.0.0)
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.2
                )

                # ===== Robust extraction of model text across SDK versions =====
                ai_text = ""
                # Preferred: new SDK object access
                try:
                    ai_text = resp.choices[0].message.content
                except Exception:
                    try:
                        # fallback for dict-like return shapes
                        ai_text = resp["choices"][0]["message"]["content"]
                    except Exception:
                        try:
                            # some older shapes
                            ai_text = resp.choices[0].text
                        except Exception:
                            # last-resort fallback: stringify full response
                            ai_text = str(resp)

                # ===== Display parsed sections =====
                st.subheader("🧾 Health Summary")
                if "Summary:" in ai_text:
                    try:
                        summary = ai_text.split("Summary:")[1].split("Questions:")[0].strip()
                        st.write(summary)
                    except Exception:
                        st.write(ai_text)
                else:
                    st.write(ai_text if len(ai_text) < 1000 else ai_text[:1000] + "...")

                st.subheader("❓ Suggested Questions for the Doctor")
                if "Questions:" in ai_text:
                    try:
                        questions = ai_text.split("Questions:")[1].split("Nutrition:")[0].strip()
                        st.write(questions)
                    except Exception:
                        st.write("Questions section parse error; see full output below.")
                else:
                    st.write("No clearly labeled 'Questions:' section detected. See full output below.")

                st.subheader("🥗 Nutrition Recommendations")
                if "Nutrition:" in ai_text:
                    try:
                        nutrition = ai_text.split("Nutrition:")[1].strip()
                        st.write(nutrition)
                    except Exception:
                        st.write("Nutrition section parse error; see full output below.")
                else:
                    st.write("No clearly labeled 'Nutrition:' section detected. See full output below.")

                # Expandable raw output for debugging and review
                with st.expander("Full AI output (raw)"):
                    st.code(ai_text)

            except Exception as e:
                # Show clear error message for debugging
                st.error(f"OpenAI API call failed: {e}")
                # Optionally print more detail to the logs (helpful during debugging)
                st.write("---")
                st.write("If this is an authentication or usage error, check the following:")
                st.write("- OPENAI_API_KEY is set in Streamlit Secrets and is valid.")
                st.write("- Your OpenAI account has quota or permission to use the model.")
                st.write("- See Streamlit app logs for deploy/build errors if any.")







