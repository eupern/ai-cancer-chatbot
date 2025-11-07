# app.py
import streamlit as st
import os
import openai
from PIL import Image
import io

st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")

st.title("🧠 AI-Driven Personalized Cancer Care Chatbot")
st.write("Upload a medical report or paste a short test/result. Click Generate to get: health summary, doctor questions, and tailored nutrition advice.")

# Read OpenAI API key from Streamlit secrets or environment
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Add OPENAI_API_KEY in Streamlit Secrets.")
    api_key_input = st.text_input("Or paste your OpenAI API key for this session (will not be saved to GitHub):", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Input area
st.subheader("1) Input report or summary")
uploaded_file = st.file_uploader("Upload a medical report (JPG/PNG/PDF) — optional for now", type=["jpg","jpeg","png","pdf"])
text_input = st.text_area("Or paste a short report / lab excerpt here (e.g., 'WBC low, Hb normal, fasting glucose 7.2 mmol/L')", height=160)

# If an image is uploaded, show preview (OCR will be added later)
if uploaded_file:
    try:
        # For image preview. If PDF, try to show first page as image (simple)
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded report (preview)", use_column_width=True)
        else:
            st.write("Uploaded file received (preview not available for PDFs in this demo).")
    except Exception as e:
        st.write("Preview error:", e)

# Use text_input if provided; else None
input_source = text_input.strip() if text_input.strip() else None

# Action button
if st.button("Generate Summary & Recommendations"):
    if not input_source:
        st.error("Please paste a short report excerpt in the text box (or enable OCR).")
    elif not OPENAI_API_KEY:
        st.error("OpenAI API key not configured. Add it to Streamlit Secrets and redeploy.")
    else:
        with st.spinner("Generating... (this may take a few seconds)"):
            # Prompt: instruct model to return clear sections
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
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.2
                )
                ai_text = resp["choices"][0]["message"]["content"]
                # Display results by section
                st.subheader("🧾 Health Summary")
                # naive parsing by labels
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





