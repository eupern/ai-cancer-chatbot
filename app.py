import streamlit as st
import os
import re
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)

st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("AI-Driven Personalized Cancer Care Chatbot")
st.write("Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, nutrition advice.")

# OpenAI key setup
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    api_key_input = st.text_input("OpenAI API Key (session only):", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Upload / paste
uploaded_files = st.file_uploader("Upload medical reports / imaging files (JPG/PNG/PDF). You can upload multiple files.", type=["jpg","jpeg","png","pdf"], accept_multiple_files=True)
text_input = st.text_area("Or paste a short lab/test excerpt here (English preferred)", height=160)

lab_texts = []
image_texts = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type.startswith("image"):
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption=f"Preview: {uploaded_file.name}", use_column_width=True)
                ocr_result = "\n".join(reader.readtext(np.array(img), detail=0))
            elif uploaded_file.type == "application/pdf":
                pages = convert_from_bytes(uploaded_file.read())
                ocr_result = ""
                for page in pages:
                    ocr_result += "\n".join(reader.readtext(np.array(page.convert("RGB")), detail=0)) + "\n"
            else:
                continue

            fname_lower = uploaded_file.name.lower()
            if any(k in fname_lower for k in ["pet", "ct", "xray", "scan"]):
                image_texts.append(ocr_result)
            else:
                lab_texts.append(ocr_result)
        except Exception as e:
            st.error(f"OCR failed for {uploaded_file.name}: {e}")

if lab_texts:
    text_input = st.text_area("OCR extracted lab text (editable, English recommended)", value="\n".join(lab_texts), height=200)

input_source = text_input.strip() if text_input and text_input.strip() else None

# Health index helpers
def compute_health_index_smart(report_text):
    score = 50
    positives = ["normal", "stable", "remission", "improved"]
    negatives = ["metastasis", "high", "low", "elevated", "decreased", "critical", "abnormal", "progression"]
    t = report_text.lower()
    for p in positives:
        if p in t:
            score += 5
    for n in negatives:
        if n in t:
            score -= 5
    return max(0, min(100, score))

def compute_health_index_with_imaging(report_texts, image_reports_texts=None):
    lab_score = compute_health_index_smart(report_texts) if report_texts else 50
    image_score = 100
    if image_reports_texts:
        deductions = 0
        for text in image_reports_texts:
            t = text.lower()
            if any(k in t for k in ["metastasis", "lesion", "tumor growth", "progression"]):
                deductions += 10
            if any(k in t for k in ["stable", "no abnormality", "remission", "no evidence of disease"]):
                deductions -= 5
        image_score = max(0, min(100, image_score - deductions))
    combined = lab_score * 0.7 + image_score * 0.3
    return round(combined, 1)

# Lab parser
def parse_lab_values(text):
    if not text:
        return {}
    t = text.lower()
    results = {}
    def find_one(patterns):
        for p in patterns:
            m = re.search(p, t, flags=re.IGNORECASE)
            if m:
                for g in range(1, 4):
                    try:
                        val = m.group(g)
                        if val:
                            val = val.replace(",", "").strip()
                            num = re.search(r"[-+]?\d*\.?\d+", val)
                            if num:
                                return float(num.group(0))
                    except Exception:
                        continue
        return None

    raw_hb = find_one([r"hemoglobin[:\s]*([\d\.]+)", r"hgb[:\s]*([\d\.]+)"])
    raw_wbc = find_one([r"wbc[:\s]*([\d\.]+)", r"white blood cell[s]?:[:\s]*([\d\.]+)", r"wbc count[:\s]*([\d\.]+)"])

    wbc_10e9 = raw_wbc / 1000.0 if raw_wbc and raw_wbc > 50 else raw_wbc

    results['hb_g_dl'] = raw_hb
    results['wbc_10e9_per_L'] = round(wbc_10e9, 2) if wbc_10e9 else None
    return results

# Dietary deep dive
def generate_dietary_deep_dive_en(lab_vals):
    lines = []
    wbc = lab_vals.get('wbc_10e9_per_L')
    if wbc and wbc < 3:
        lines.append("Patient shows low WBC. Follow strict food safety and well-cooked diet.")
        lines.append("- Cooked proteins (eggs, fish, chicken, tofu)")
        lines.append("- Cooked whole grains and fiber-rich foods")
        lines.append("- Pasteurized dairy, avoid raw foods")
    else:
        lines.append("No critical lab flags. Maintain balanced nutrition with cooked meals, vegetables, and proteins.")
    return "\n".join(lines)

# Generate summary
if st.button("Generate Summary & Recommendations"):
    if not input_source and not lab_texts:
        st.error("Please paste a lab/test excerpt or upload OCR-compatible files first.")
    elif not client:
        st.error("OpenAI client not configured.")
    else:
        with st.spinner("Generating AI output..."):
            all_lab_text = input_source if input_source else "\n".join(lab_texts)
            health_index = compute_health_index_with_imaging(all_lab_text, image_texts)
            st.subheader("Health Index")
            st.write(f"Combined Health Index (0-100): {health_index}")

            prompt = f"""You are a clinical-support assistant. Respond only in English. Produce exactly three labelled sections: Summary, Questions, Nutrition. Use plain English and provide dietitian-level advice based on Malaysia Cancer Nutrition Guidelines. Patient report:\n{all_lab_text}"""

            try:
                resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":prompt}], max_tokens=700, temperature=0.2)
                ai_text = resp.choices[0].message.content
                st.subheader("AI Output")
                st.text_area("AI Summary", value=ai_text, height=400)

                labs = parse_lab_values(all_lab_text)
                deep_text = generate_dietary_deep_dive_en(labs)
                st.subheader("Dietary Advice (English)")
                st.text_area("Dietary Deep Dive", value=deep_text, height=300)

                st.subheader("Optional: Send report to email")
                email_address = st.text_input("Enter email to send report (optional)")
                if email_address and st.button("Send Email" + email_address):
                    st.info(f"Simulated sending report to {email_address} — implement your email backend here.")

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")































