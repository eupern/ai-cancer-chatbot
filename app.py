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
from email.mime.text import MIMEText

# ---------------------------
# OCR Setup
# ---------------------------
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("AI-Driven Personalized Cancer Care Chatbot")
st.write("Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, and nutrition advice.")

# ---------------------------
# OpenAI Client Setup
# ---------------------------
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

# ---------------------------
# Input: File Upload or Lab Text
# ---------------------------
st.subheader("1) Input medical reports or lab summary")
uploaded_files = st.file_uploader(
    "Upload medical reports / imaging files (JPG/PNG/PDF). You can upload multiple files.",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)
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

# ---------------------------
# Health Index Helpers
# ---------------------------
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

# ---------------------------
# Lab Parsing
# ---------------------------
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
                    except Exception:
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
            wbc_note = f"converted from {raw_wbc} (assumed cells/µL) to {wbc_10e9:.2f} x10^9/L"
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
            neut_note = f"calculated from {raw_neut_percent}% of {wbc_10e9:.2f} x10^9/L -> {neut_abs:.2f} x10^9/L"
        except:
            neut_abs = None

    results['hb_g_dl'] = raw_hb
    results['wbc_raw'] = raw_wbc
    results['wbc_10e9_per_L'] = round(wbc_10e9, 2) if wbc_10e9 is not None else None
    results['wbc_note'] = wbc_note
    results['neutrophil_abs_raw'] = raw_neut_abs
    results['neutrophil_abs'] = round(neut_abs, 2) if neut_abs is not None else None
    results['neut_note'] = neut_note
    results['neut_percent_raw'] = raw_neut_percent
    results['plt'] = raw_plt
    results['glucose'] = raw_glu
    return results

# ---------------------------
# Dietary Deep Dive (always dietitian-level)
# ---------------------------
def generate_dietary_deep_dive(lab_vals):
    lines = []
    lines.append("Dietitian-Level Dietary Guidance:")
    wbc_note = lab_vals.get('wbc_note')
    neut_note = lab_vals.get('neut_note')
    if wbc_note:
        lines.append(f"(Note: {wbc_note})")
    if neut_note:
        lines.append(f"(Note: {neut_note})")

    lines.extend([
        "- Ensure adequate high-quality protein from cooked meats, fish, eggs, tofu, pasteurised dairy.",
        "- Include cooked whole grains, fiber, and vegetables to support gut and immunity.",
        "- Maintain hydration and follow strict food safety (avoid raw or undercooked foods).",
        "- Include antioxidant-rich foods in moderate amounts; follow physician advice for supplements.",
        "- Balance meals to support energy, immunity, and recovery."
    ])
    lines.append("Sample 1-day menu (reference):")
    lines.extend([
        "- Breakfast: cooked oats + banana + pumpkin seeds + pasteurised yogurt",
        "- Lunch: steamed chicken breast + brown rice + steamed vegetables",
        "- Dinner: baked fish + quinoa + steamed greens",
        "- Snacks: hard-boiled egg, cooked fruit compote"
    ])
    return "\n".join(lines)

# ---------------------------
# Extract Sections from AI
# ---------------------------
def extract_section(text, header):
    pattern = rf"{header}\s*[:\-]?\s*(.*?)(?=\n(?:Summary|Questions|Nutrition)\s*[:\-]|\Z)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    variants = {
        "Questions": [r"questions to ask", r"doctor questions", r"questions:", r"questions to ask the doctor", r"questions for doctor"],
        "Summary": [r"summary", r"health summary", r"clinical summary"],
        "Nutrition": [r"nutrition", r"recommendations", r"diet", r"nutrition recommendations"]
    }
    for v in variants.get(header, []):
        m2 = re.search(rf"{v}\s*[:\-]?\s*(.*?)(?=\n(?:summary|questions|nutrition)\s*[:\-]|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        if m2:
            return m2.group(1).strip()
    return "No findings."

# ---------------------------
# Generate AI Output
# ---------------------------
if st.button("Generate Summary & Recommendations"):
    if not input_source and not lab_texts:
        st.error("Please paste a lab/test excerpt or upload OCR-compatible files first.")
    elif not client:
        st.error("OpenAI client not configured. Please set OPENAI_API_KEY.")
    else:
        with st.spinner("Generating AI output..."):
            all_lab_text = input_source if input_source else "\n".join(lab_texts)
            health_index = compute_health_index_with_imaging(all_lab_text, image_texts)
            st.session_state['health_index'] = health_index
            st.subheader("Health Index")
            st.write(f"Combined Health Index (0-100): {health_index}")

            prompt = f"""You are a clinical-support assistant. Respond only in English.
Produce exactly three labelled sections: Summary, Questions, Nutrition.
- Write "Summary:" then 3-4 short sentences in plain English.
- Write "Questions:" then a numbered or bulleted list of three practical questions for the patient/family.
- Write "Nutrition:" then three food-based nutrition recommendations based on Malaysia Cancer Nutrition Guidelines.
Patient report:
\"\"\"{all_lab_text}\"\"\"
Required output headers (exactly): 
Summary:
- ...
Questions:
- ...
Nutrition:
- ...
"""
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=700,
                    temperature=0.2
                )
                ai_text = resp.choices[0].message.content
                summary = extract_section(ai_text, "Summary")
                questions = extract_section(ai_text, "Questions")
                nutrition = extract_section(ai_text, "Nutrition")

                st.subheader("Health Summary")
                st.write(summary)
                st.subheader("Suggested Questions for the Doctor")
                st.write(questions)
                st.subheader("Nutrition Recommendations")
                st.write(nutrition)

                # Lab parse and dietary deep dive
                labs = parse_lab_values(all_lab_text)
                st.session_state['labs'] = labs
                deep_text = generate_dietary_deep_dive(labs)
                st.subheader("Dietary Deep Dive (English, dietitian-level)")
                st.text_area("Dietary Deep Dive", value=deep_text, height=320)

                # Optional: ask more questions to refine advice
                st.subheader("Ask more questions / refine advice")
                followup_q = st.text_area("Enter your follow-up questions for AI refinement:", height=100)
                if st.button("Refine Advice"):
                    if followup_q.strip():
                        prompt_refine = f"""Patient original report:
\"\"\"{all_lab_text}\"\"\"
Dietary Deep Dive already provided:
\"\"\"{deep_text}\"\"\"
Patient/family follow-up question: {followup_q.strip()}
Please refine and provide updated Summary, Questions, Nutrition, and Dietitian-level guidance. Respond only in English."""
                        resp2 = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role":"user","content":prompt_refine}],
                            max_tokens=700,
                            temperature=0.2
                        )
                        ai_text2 = resp2.choices[0].message.content
                        st.subheader("Refined AI Summary & Recommendations")
                        st.text_area("Refined AI output", value=ai_text2, height=300)

                # Optional email column
                st.subheader("Send Final Report via Email (optional)")
                send_email = st.checkbox("Send report to email")
                if send_email:
                    email_address = st.text_input("Recipient email address")
                    if st.button("Send Email"):
                        try:
                            content = f"Health Index: {health_index}\n\nSummary:\n{summary}\n\nQuestions:\n{questions}\n\nNutrition:\n{nutrition}\n\nDietary Deep Dive:\n{deep_text}"
                            msg = MIMEText(content)
                            msg['Subject'] = "AI-Driven Personalized Health Report"
                            msg['From'] = "noreply@example.com"
                            msg['To'] = email_address
                            # Example SMTP send (configure your SMTP)
                            s = smtplib.SMTP('localhost')
                            s.send_message(msg)
                            s.quit()
                            st.success(f"Report sent to {email_address}")
                        except Exception as e:
                            st.error(f"Failed to send email: {e}")

                with st.expander("Full AI output (raw)"):
                    st.code(ai_text)

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")




































