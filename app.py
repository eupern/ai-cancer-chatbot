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
from email.mime.multipart import MIMEMultipart

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)

# Streamlit config
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("AI-Driven Personalized Cancer Care Chatbot")
st.write(
    "Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, and dietitian-level dietary advice."
)

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

# Session state initialization
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'lab_texts' not in st.session_state:
    st.session_state.lab_texts = []
if 'image_texts' not in st.session_state:
    st.session_state.image_texts = []
if 'all_lab_text' not in st.session_state:
    st.session_state.all_lab_text = ""
if 'health_index' not in st.session_state:
    st.session_state.health_index = None
if 'deep_dive' not in st.session_state:
    st.session_state.deep_dive = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'questions' not in st.session_state:
    st.session_state.questions = ""
if 'ai_raw' not in st.session_state:
    st.session_state.ai_raw = ""

# File upload and text input
st.subheader("1) Input medical reports or lab summary")
uploaded_files = st.file_uploader(
    "Upload medical reports / imaging files (JPG/PNG/PDF). You can upload multiple files.",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True,
    key="file_uploader"
)

text_input = st.text_area(
    "Or paste a short lab/test excerpt here (English preferred)", 
    value="\n".join(st.session_state.lab_texts),
    height=160
)

# Preserve uploaded files in session_state
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# OCR extraction (only if first upload or new files)
if uploaded_files and not st.session_state.lab_texts:
    lab_texts = []
    image_texts = []
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
        st.session_state.lab_texts = lab_texts
    if image_texts:
        st.session_state.image_texts = image_texts

# Input source
input_source = text_input.strip() if text_input and text_input.strip() else "\n".join(st.session_state.lab_texts)
st.session_state.all_lab_text = input_source

# Health index functions
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
        if raw_wbc > 50:
            wbc_10e9 = raw_wbc / 1000.0
            wbc_note = f"converted from {raw_wbc} (assumed cells/µL) to {wbc_10e9:.2f} x10^9/L"
        else:
            wbc_10e9 = raw_wbc
            wbc_note = f"assumed reported in 10^9/L: {wbc_10e9:.2f} x10^9/L"

    neut_abs = None
    neut_note = None
    if raw_neut_abs is not None:
        if raw_neut_abs > 50:
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

# Dietitian-level deep dive
def generate_dietary_deep_dive_en(lab_vals, extra_question=None):
    lines = []
    lines.append("Clinical note & overview:")
    wbc_note = lab_vals.get('wbc_note')
    neut_note = lab_vals.get('neut_note')
    if wbc_note:
        lines.append(f"(Note: {wbc_note})")
    if neut_note:
        lines.append(f"(Note: {neut_note})")

    wbc = lab_vals.get('wbc_10e9_per_L') or lab_vals.get('wbc_raw')
    neut = lab_vals.get('neutrophil_abs')
    hb = lab_vals.get('hb_g_dl')
    plt = lab_vals.get('plt')
    glu = lab_vals.get('glucose')

    neutropenia_flag = False
    severity = None
    if neut is not None:
        if neut < 1.5:
            neutropenia_flag = True
            if neut < 0.5:
                severity = "severe"
            elif neut < 1.0:
                severity = "moderate"
            else:
                severity = "mild"
    elif wbc is not None:
        try:
            w = float(wbc)
            if w < 3.0:
                neutropenia_flag = True
                severity = "possible (WBC low)"
        except:
            pass

    if neutropenia_flag:
        lines.append(f"Patient shows neutropenia / low WBC (severity: {severity}). Infection risk is elevated — follow food-safety measures and dietary guidance.")
        lines.append("Practical guidance (neutropenia):")
        lines.extend([
            "- Prioritise well-cooked, high-quality proteins: fully cooked eggs, cooked fish, skinless chicken, tofu, pasteurised yogurt.",
            "- Include cooked whole grains and soluble fiber: cooked oats, brown rice, cooked banana, oat bran.",
            "- Avoid raw seafood, salads, sprouts, undercooked meats.",
            "- Include zinc/selenium foods (pumpkin seeds, small Brazil nuts).",
            "- Maintain hydration and oral hygiene; seek care if fever occurs."
        ])
        lines.append("Nutritional rationale:")
        lines.append("- Cooked whole grains and soluble fiber feed the gut microbiome to produce SCFA, supporting gut barrier and immune resilience.")
        lines.append("- High-quality protein supports tissue repair and immune cell synthesis.")

    if hb is not None and hb < 12:
        lines.append("Anemia-related suggestions:")
        lines.append("- Increase iron and quality protein sources (lean red meat, legumes, spinach, pumpkin seeds) paired with vitamin C.")

    if plt is not None and plt < 100:
        lines.append("Low platelets (bleeding risk) notes:")
        lines.append("- Avoid hard, sharp foods; modify texture (chop or grind nuts).")

    if glu is not None:
        if glu > 7.0:
            lines.append("Glucose notes: Hyperglycaemia present — reduce refined sugars/drinks; prefer whole grains and vegetables.")
        else:
            lines.append("Glucose notes: within acceptable range; maintain balanced carbs and protein.")

    lines.append("Sample 1-day menu (reference):")
    lines.extend([
        "- Breakfast: cooked oats + cooked banana + pumpkin seeds + pasteurised yogurt.",
        "- Lunch: steamed chicken breast + cooked brown rice + steamed carrot + cooked spinach.",
        "- Dinner: steamed fish + cooked quinoa/brown rice + steamed greens.",
        "- Snacks: hard-boiled egg, small portion cooked fruit compote."
    ])

    if extra_question:
        lines.append("\nRefined advice based on user follow-up question:")
        lines.append(f"- {extra_question}")

    return "\n".join(lines)

# Section extractor
def extract_section(text, header):
    pattern = rf"{header}\s*[:\-]?\s*(.*?)(?=\n(?:Summary|Questions|Nutrition|Dietary Deep Dive)\s*[:\-]|\Z)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return "No findings."

# -----------------------------
# Main buttons
# -----------------------------
st.subheader("2) Generate Health Summary & Dietary Advice")
if st.button("Generate Summary & Deep Dive"):
    if not input_source:
        st.error("Please paste a lab/test excerpt or upload files first.")
    elif not client:
        st.error("OpenAI client not configured.")
    else:
        with st.spinner("Generating AI output..."):
            all_lab_text = st.session_state.all_lab_text
            st.session_state.health_index = compute_health_index_with_imaging(all_lab_text, st.session_state.image_texts)
            st.subheader("Health Index")
            st.write(f"Combined Health Index (0-100): {st.session_state.health_index}")

            # AI prompt for summary & questions
            prompt = f"""
You are a clinical-support assistant. Respond only in English.
Given the patient's report text below, produce exactly three labelled sections: Summary, Questions, Dietary Deep Dive.
- Summary: 3-4 short sentences in plain, simple English.
- Questions: 3 practical questions the patient/family should ask the doctor next visit.
- Dietary Deep Dive: dietitian-level, structured guidance including sample menu, rationale, and food-safety notes. 
Patient report:
\"\"\"{all_lab_text}\"\"\"
"""

            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=1000,
                    temperature=0.2
                )
                ai_text = resp.choices[0].message.content
                st.session_state.ai_raw = ai_text

                # Extract sections
                st.session_state.summary = extract_section(ai_text, "Summary")
                st.session_state.questions = extract_section(ai_text, "Questions")
                st.session_state.deep_dive = extract_section(ai_text, "Dietary Deep Dive")

                st.subheader("Health Summary")
                st.write(st.session_state.summary)
                st.subheader("Suggested Questions for the Doctor")
                st.write(st.session_state.questions)
                st.subheader("Dietitian-level Dietary Advice")
                st.text_area("Dietary Deep Dive (copyable)", value=st.session_state.deep_dive, height=350)

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")

# Follow-up question to refine advice
st.subheader("3) Refine / Ask More Questions")
follow_up = st.text_area("Ask a follow-up question or clarify for dietary advice:", height=80)
if st.button("Refine Advice"):
    if not st.session_state.deep_dive:
        st.error("Generate initial summary first.")
    else:
        refined_text = generate_dietary_deep_dive_en(parse_lab_values(st.session_state.all_lab_text), extra_question=follow_up)
        st.session_state.deep_dive = refined_text
        st.text_area("Dietary Deep Dive (copyable)", value=st.session_state.deep_dive, height=350)

# Optional email sending
st.subheader("4) Send Final Report via Email (Optional)")
email_to = st.text_input("Recipient email (optional)")
if st.button("Send Email"):
    if not email_to:
        st.error("Please enter recipient email.")
    else:
        try:
            sender_email = st.secrets.get("SENDER_EMAIL")
            sender_pass = st.secrets.get("SENDER_PASSWORD")
            if not sender_email or not sender_pass:
                st.error("Email credentials not set in secrets.")
            else:
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = email_to
                msg['Subject'] = "Personalized Cancer Care Report"
                body = f"Health Index: {st.session_state.health_index}\n\nSummary:\n{st.session_state.summary}\n\nQuestions:\n{st.session_state.questions}\n\nDietary Advice:\n{st.session_state.deep_dive}"
                msg.attach(MIMEText(body, 'plain'))
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(sender_email, sender_pass)
                server.send_message(msg)
                server.quit()
                st.success(f"Report sent to {email_to}")
        except Exception as e:
            st.error(f"Failed to send email: {





































