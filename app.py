# app.py
import streamlit as st
import os
import re
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr
import smtplib
from email.message import EmailMessage

# Initialize EasyOCR
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)

# Streamlit config
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("AI-Driven Personalized Cancer Care Chatbot")
st.write("Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, and dietitian-level dietary advice.")

# OpenAI API key
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

# File upload / text input
st.subheader("1) Input medical reports or lab summary")
uploaded_files = st.file_uploader(
    "Upload medical reports / imaging files (JPG/PNG/PDF). You can upload multiple files.",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)
text_input = st.text_area("Or paste a short lab/test excerpt here (English preferred)", height=160)

# OCR extraction
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
                    page_arr = np.array(page.convert("RGB"))
                    ocr_result += "\n".join(reader.readtext(page_arr, detail=0)) + "\n"
            else:
                st.warning(f"{uploaded_file.name} is not a supported file type.")
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

# Lab parsing
def parse_lab_values(text):
    if not text:
        return {}
    t = text.lower()
    results = {}
    def find_one(patterns):
        for p in patterns:
            m = re.search(p, t, flags=re.IGNORECASE)
            if m:
                for g in (1,2,3):
                    try:
                        val = m.group(g)
                        if val:
                            val = val.replace(",", "").strip()
                            num = re.search(r"[-+]?\d*\.?\d+", val)
                            if num:
                                return float(num.group(0))
                    except:
                        continue
        return None

    raw_hb = find_one([r"hemoglobin[:\s]*([\d\.]+)", r"hgb[:\s]*([\d\.]+)"])
    raw_wbc = find_one([r"wbc[:\s]*([\d\.]+)", r"white blood cell[s]?:[:\s]*([\d\.]+)", r"wbc count[:\s]*([\d\.]+)"])
    raw_neut_abs = find_one([r"neutrophil[s]?\s*(?:absolute)?[:\s]*([\d\.]+)", r"neutrophil count[:\s]*([\d\.]+)"])
    raw_neut_percent = find_one([r"neutrophil[s]?\s*%\s*[:\s]*([\d\.]+)", r"neutrophil[s]?\s*percent[:\s]*([\d\.]+)"])
    raw_plt = find_one([r"platelet[s]?:[:\s]*([\d\.]+)", r"plt[:\s]*([\d\.]+)"])
    raw_glu = find_one([r"glucose[:\s]*([\d\.]+)", r"fasting glucose[:\s]*([\d\.]+)"])

    wbc_10e9 = None
    wbc_note = None
    if raw_wbc is not None:
        if raw_wbc > 1000 or raw_wbc > 50:
            wbc_10e9 = raw_wbc / 1000.0
            wbc_note = f"converted from {raw_wbc} to {wbc_10e9:.2f} x10^9/L"
        else:
            wbc_10e9 = raw_wbc
            wbc_note = f"assumed reported in 10^9/L: {wbc_10e9:.2f} x10^9/L"

    neut_abs = None
    neut_note = None
    if raw_neut_abs is not None:
        if raw_neut_abs > 1000 or raw_neut_abs > 50:
            neut_abs = raw_neut_abs / 1000.0
            neut_note = f"converted from {raw_neut_abs} to {neut_abs:.2f} x10^9/L"
        else:
            neut_abs = raw_neut_abs
            neut_note = f"assumed in 10^9/L: {neut_abs:.2f} x10^9/L"
    elif raw_neut_percent is not None and wbc_10e9 is not None:
        try:
            neut_abs = (raw_neut_percent / 100.0) * wbc_10e9
            neut_note = f"calculated from {raw_neut_percent}% of {wbc_10e9:.2f} -> {neut_abs:.2f} x10^9/L"
        except:
            neut_abs = None

    results['hb_g_dl'] = raw_hb
    results['wbc_raw'] = raw_wbc
    results['wbc_10e9_per_L'] = round(wbc_10e9, 2) if wbc_10e9 else None
    results['wbc_note'] = wbc_note
    results['neutrophil_abs_raw'] = raw_neut_abs
    results['neutrophil_abs'] = round(neut_abs, 2) if neut_abs else None
    results['neut_note'] = neut_note
    results['neut_percent_raw'] = raw_neut_percent
    results['plt'] = raw_plt
    results['glucose'] = raw_glu
    return results

# Dietary advice generator (always dietitian-level)
def generate_dietary_advice(lab_vals):
    lines = []
    lines.append("Dietary advice (dietitian-level):")
    wbc_note = lab_vals.get('wbc_note')
    neut_note = lab_vals.get('neut_note')
    if wbc_note: lines.append(f"(Note: {wbc_note})")
    if neut_note: lines.append(f"(Note: {neut_note})")

    # Neutropenia / low WBC guidance
    neut = lab_vals.get('neutrophil_abs')
    wbc = lab_vals.get('wbc_10e9_per_L') or lab_vals.get('wbc_raw')
    if neut is not None and neut < 1.5:
        lines.append("- Patient shows low neutrophils / WBC. Follow strict food-safety and hygiene measures.")
    elif wbc is not None and wbc < 3:
        lines.append("- WBC slightly low. Emphasize safe cooked meals.")

    # General nutrition
    lines.append("- Include well-cooked high-quality proteins: eggs, fish, chicken, tofu, pasteurized yogurt.")
    lines.append("- Cooked whole grains and vegetables for fiber and gut health.")
    lines.append("- Nuts, seeds, and small amounts of healthy oils (olive, avocado) for micronutrients and healthy fats.")

    lines.append("\nSample 1-day menu (reference):")
    lines.append("- Breakfast: cooked oats + cooked banana + pumpkin seeds + pasteurized yogurt")
    lines.append("- Lunch: steamed chicken breast + cooked brown rice + steamed carrot + cooked spinach")
    lines.append("- Dinner: steamed fish + cooked quinoa + steamed greens")
    lines.append("- Snacks: hard-boiled egg, small portion cooked fruit compote")
    return "\n".join(lines)

# Robust extractor for GPT output sections
def extract_section(text, header):
    pattern = rf"{header}\s*[:\-]?\s*(.*?)(?=\n(?:Summary|Questions|Nutrition|Dietary)\s*[:\-]|\Z)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if m: return m.group(1).strip()
    return "No findings."

# Main AI generation button
if st.button("Generate Summary & Dietary Advice"):
    if not input_source and not lab_texts:
        st.error("Please paste a lab/test excerpt or upload files first.")
    elif not client:
        st.error("OpenAI client not configured. Please set OPENAI_API_KEY.")
    else:
        with st.spinner("Generating AI output..."):
            all_lab_text = input_source if input_source else "\n".join(lab_texts)
            health_index = compute_health_index_with_imaging(all_lab_text, image_texts)
            st.session_state['health_index'] = health_index
            st.subheader("Health Index")
            st.write(f"Combined Health Index (0-100): {health_index}")
            
            prompt = f"""
You are a clinical-support assistant. Respond only in English.
Given the patient's report text below, produce exactly three labelled sections: Summary, Questions, Dietary Advice.
- Summary: 3-4 short sentences in plain English.
- Questions: 3 practical questions the patient/family should ask the doctor.
- Dietary Advice: dietitian-level, with rationale and 1-day sample menu.

Patient report:
\"\"\"{all_lab_text}\"\"\"

Output headers exactly:
Summary:
Questions:
Dietary Advice:
"""
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user", "content": prompt}],
                    max_tokens=700,
                    temperature=0.2
                )
                ai_text = resp.choices[0].message.content
                summary = extract_section(ai_text, "Summary")
                questions = extract_section(ai_text, "Questions")
                dietary = extract_section(ai_text, "Dietary Advice")

                st.session_state['summary'] = summary
                st.session_state['questions'] = questions
                st.session_state['dietary'] = dietary
                st.session_state['ai_raw'] = ai_text

                st.subheader("Health Summary")
                st.write(summary)
                st.markdown("---")
                st.subheader("Suggested Questions for the Doctor")
                st.write(questions)
                st.markdown("---")
                st.subheader("Dietary Advice")
                st.text_area("Dietary Advice (dietitian-level)", value=dietary, height=320)

                # Optional follow-up question to refine
                followup_q = st.text_area("Ask follow-up questions to refine advice", height=80)
                if st.button("Refine Advice"):
                    if followup_q.strip():
                        with st.spinner("Refining advice..."):
                            followup_prompt = f"""
Patient report:
\"\"\"{all_lab_text}\"\"\"
Previous AI output:
\"\"\"{ai_text}\"\"\"
User follow-up question:
\"\"\"{followup_q}\"\"\"
Please refine the summary, questions, and dietary advice accordingly.
Respond only in English, dietitian-level advice, include 1-day menu.
"""
                            resp2 = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role":"user", "content": followup_prompt}],
                                max_tokens=700,
                                temperature=0.2
                            )
                            ai_text2 = resp2.choices[0].message.content
                            summary2 = extract_section(ai_text2, "Summary")
                            questions2 = extract_section(ai_text2, "Questions")
                            dietary2 = extract_section(ai_text2, "Dietary Advice")
                            st.session_state['summary'] = summary2
                            st.session_state['questions'] = questions2
                            st.session_state['dietary'] = dietary2
                            st.text_area("Refined Dietary Advice", value=dietary2, height=320)
            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")

# Optional: email sending
st.markdown("---")
st.subheader("Optional: Send summary via email")
send_email = st.checkbox("Send final report via email")
if send_email:
    to_email = st.text_input("Recipient Email")
    if st.button("Send Email") and to_email:
        try:
            msg = EmailMessage()
            msg['Subject'] = "Personalized Health Summary & Dietary Advice"
            msg['From'] = "your_email@example.com"
            msg['To'] = to_email
            body = (
                f"Health Index: {st.session_state.get('health_index','N/A')}\n\n"
                f"Summary:\n{st.session_state.get('summary','No findings.')}\n\n"
                f"Questions:\n{st.session_state.get('questions','No findings.')}\n\n"
                f"Dietary Advice:\n{st.session_state.get('dietary','No findings.')}"
            )
            msg.set_content(body)
            with smtplib.SMTP('localhost') as server:
                server.send_message(msg)
            st.success(f"Email sent to {to_email}")
        except Exception as e:
            st.error(f"Failed to send email: {e}")







































