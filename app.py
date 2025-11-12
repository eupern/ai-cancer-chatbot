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

# -------------------------
# Config / Initialization
# -------------------------
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)

st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("AI-Driven Personalized Cancer Care Chatbot")
st.write("Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, and dietitian-level dietary advice.")

# OpenAI client setup
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

# -------------------------
# Session state defaults
# -------------------------
if 'uploaded_files_meta' not in st.session_state:
    st.session_state['uploaded_files_meta'] = None
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = None
if 'ocr_lab_texts' not in st.session_state:
    st.session_state['ocr_lab_texts'] = []
if 'ocr_image_texts' not in st.session_state:
    st.session_state['ocr_image_texts'] = []
if 'all_lab_text' not in st.session_state:
    st.session_state['all_lab_text'] = ""
if 'ai_raw' not in st.session_state:
    st.session_state['ai_raw'] = ""
if 'summary' not in st.session_state:
    st.session_state['summary'] = ""
if 'questions' not in st.session_state:
    st.session_state['questions'] = ""
if 'dietary' not in st.session_state:
    st.session_state['dietary'] = ""
if 'generated' not in st.session_state:
    st.session_state['generated'] = False
if 'health_index' not in st.session_state:
    st.session_state['health_index'] = None

# -------------------------
# Helpers
# -------------------------
def compute_health_index_smart(report_text):
    score = 50
    positives = ["normal", "stable", "remission", "improved"]
    negatives = ["metastasis", "high", "low", "elevated", "decreased", "critical", "abnormal", "progression"]
    t = (report_text or "").lower()
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
                        v = m.group(g)
                        if v:
                            v = v.replace(",", "").strip()
                            num = re.search(r"[-+]?\d*\.?\d+", v)
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
        if raw_wbc > 50:
            wbc_10e9 = raw_wbc / 1000.0
            wbc_note = f"converted from {raw_wbc} to {wbc_10e9:.2f} x10^9/L"
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
            neut_note = f"calculated from {raw_neut_percent}% of {wbc_10e9:.2f} -> {neut_abs:.2f} x10^9/L"
        except:
            neut_abs = None

    results['hb_g_dl'] = raw_hb
    results['wbc_raw'] = raw_wbc
    results['wbc_10e9_per_L'] = round(wbc_10e9,2) if wbc_10e9 is not None else None
    results['wbc_note'] = wbc_note
    results['neutrophil_abs_raw'] = raw_neut_abs
    results['neutrophil_abs'] = round(neut_abs,2) if neut_abs is not None else None
    results['neut_note'] = neut_note
    results['neut_percent_raw'] = raw_neut_percent
    results['plt'] = raw_plt
    results['glucose'] = raw_glu
    return results

def generate_dietary_advice_from_labs(lab_vals):
    lines = []
    lines.append("Dietary Advice (dietitian-level):")
    if lab_vals.get('wbc_note'):
        lines.append(f"({lab_vals.get('wbc_note')})")
    if lab_vals.get('neut_note'):
        lines.append(f"({lab_vals.get('neut_note')})")
    if lab_vals.get('neutrophil_abs') is not None and lab_vals['neutrophil_abs'] < 1.5:
        lines.append("- Low neutrophils: avoid raw or undercooked foods; prioritise thoroughly cooked proteins and strict hygiene.")
    lines.append("\nSample 1-day menu (reference):")
    lines.extend([
        "- Breakfast: cooked oats + cooked banana + pumpkin seeds + pasteurised yogurt",
        "- Lunch: steamed chicken breast + cooked brown rice + steamed carrot + cooked spinach",
        "- Dinner: steamed fish + cooked quinoa/brown rice + steamed greens",
        "- Snacks: hard-boiled egg, small portion cooked fruit compote"
    ])
    lines.append("\nNutritional rationale:")
    lines.extend([
        "- High-quality protein supports tissue repair and immune function.",
        "- Cooked vegetables and grains improve digestibility and reduce infection risk."
    ])
    lines.append("\nFood safety notes:")
    lines.extend([
        "- Avoid raw seafood, raw milk, raw sprouts; prefer pasteurised dairy and well-cooked meats.",
        "- Maintain good hand/food hygiene; refrigerate perishables promptly."
    ])
    return "\n".join(lines)

def extract_section(text, header):
    pattern = rf"{header}\s*[:\-]?\s*(.*?)(?=\n(?:Summary|Questions|Dietary Advice|Nutrition)\s*[:\-]|\Z)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else "No findings."

def clean_questions_text(q_text):
    if not q_text:
        return q_text
    keywords = ['diet', 'nutrition', 'food', 'menu', 'meal', 'calorie', 'protein', 'carb', 'sugar', 'supplement']
    filtered = []
    for line in q_text.splitlines():
        low = line.lower()
        if any(k in low for k in keywords):
            continue
        filtered.append(line)
    cleaned = "\n".join(filtered).strip()
    return cleaned if cleaned else q_text

# -------------------------
# UI: Upload + OCR (persisted)
# -------------------------
st.subheader("1) Input medical reports or lab summary")

widget_files = st.file_uploader(
    "Upload medical reports / imaging files (JPG/PNG/PDF). You can upload multiple files.",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True,
    key="file_uploader"
)

if widget_files:
    st.session_state['uploaded_files'] = widget_files

curr_meta = tuple([f.name for f in st.session_state['uploaded_files']]) if st.session_state.get('uploaded_files') else None
if curr_meta and curr_meta != st.session_state.get('uploaded_files_meta'):
    lab_texts_local = []
    image_texts_local = []
    for f in st.session_state['uploaded_files']:
        try:
            if f.type.startswith("image"):
                img = Image.open(f).convert("RGB")
                st.image(img, caption=f"Preview: {f.name}", use_column_width=True)
                ocr_result = "\n".join(reader.readtext(np.array(img), detail=0))
            elif f.type == "application/pdf":
                pages = convert_from_bytes(f.read())
                ocr_result = ""
                for p in pages:
                    ocr_result += "\n".join(reader.readtext(np.array(p.convert("RGB")), detail=0)) + "\n"
            else:
                continue
            name = f.name.lower()
            if any(k in name for k in ["pet", "ct", "xray", "scan"]):
                image_texts_local.append(ocr_result)
            else:
                lab_texts_local.append(ocr_result)
        except Exception as e:
            st.error(f"OCR failed for {f.name}: {e}")
    st.session_state['ocr_lab_texts'] = lab_texts_local
    st.session_state['ocr_image_texts'] = image_texts_local
    st.session_state['uploaded_files_meta'] = curr_meta

initial_lab_text = "\n".join(st.session_state['ocr_lab_texts']) if st.session_state['ocr_lab_texts'] else ""
text_input = st.text_area("Or paste a short lab/test excerpt here (English preferred)", value=initial_lab_text, height=160)
input_source = text_input.strip() if text_input and text_input.strip() else ""
if input_source:
    st.session_state['all_lab_text'] = input_source

# -------------------------
# Main: Generate Summary & Dietary Advice
# -------------------------
st.subheader("2) Generate Summary & Dietary Advice")
if st.button("Generate Summary & Dietary Advice"):
    if not st.session_state.get('all_lab_text'):
        st.error("Please paste lab text or upload files first.")
    elif not client:
        st.error("OpenAI client not configured.")
    else:
        all_lab_text = st.session_state['all_lab_text']
        st.session_state['health_index'] = compute_health_index_with_imaging(all_lab_text, st.session_state.get('ocr_image_texts', []))
        with st.spinner("Generating AI summary..."):
            prompt = f"""
You are a clinical-support assistant. Respond only in English.
Produce exactly two labelled sections: Summary, Questions.
DO NOT include nutrition, diet, menu, or food advice in your output — the app provides dietary advice separately.

- Summary: 3-4 short sentences in plain English.
- Questions: 3 practical, clinical questions the patient/family should ask the doctor (do not ask about diet/nutrition).

Patient report:
\"\"\"{all_lab_text}\"\"\"
"""
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=700,
                    temperature=0.2
                )
                ai_text = resp.choices[0].message.content
                st.session_state['ai_raw'] = ai_text

                raw_summary = extract_section(ai_text, "Summary")
                raw_questions = extract_section(ai_text, "Questions")
                cleaned_questions = clean_questions_text(raw_questions)

                st.session_state['summary'] = raw_summary
                st.session_state['questions'] = cleaned_questions

                lab_vals = parse_lab_values(all_lab_text)
                st.session_state['dietary'] = generate_dietary_advice_from_labs(lab_vals)

                st.session_state['generated'] = True
                st.success("Summary & dietary advice generated.")
            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")

# -------------------------
# Display generated outputs (persisted)
# -------------------------
if st.session_state.get('generated'):
    st.subheader("Health Index")
    st.write(f"Combined Health Index (0-100): {st.session_state.get('health_index','N/A')}")
    st.markdown("---")
    st.subheader("Health Summary")
    st.write(st.session_state.get('summary','No findings.'))  # only summary content shown here
    st.markdown("---")
    st.subheader("Suggested Questions for the Doctor")
    st.write(st.session_state.get('questions','No findings.'))
    st.markdown("---")
    st.subheader("Dietary Advice")
    st.text_area("Dietary Advice (dietitian-level, copyable)", value=st.session_state.get('dietary',''), height=360)

# -------------------------
# Follow-up: what to refine?
# -------------------------
st.subheader("3) Ask follow-up / refine advice (optional)")
refine_choice = st.selectbox("Refine which part?", ["Dietary Advice", "Suggested Questions for the Doctor", "Both (Summary + Questions + Dietary Advice)"])
follow_up_q = st.text_area("Type follow-up question (optional):", height=90)

if st.button("Refine Selected"):
    if not st.session_state.get('generated'):
        st.error("Generate initial summary first.")
    else:
        all_lab_text = st.session_state.get('all_lab_text','')
        if not all_lab_text:
            st.error("No lab text found in session.")
        else:
            with st.spinner("Refining..."):
                # Build prompt according to selection
                if refine_choice == "Dietary Advice":
                    # Request only a Diet section from the model (we will save it under dietary)
                    prompt_ref = f"""
You are a clinical dietitian-level assistant. Respond only in English.
Given the patient report below, produce a labelled section: Dietary Advice.
- Dietary Advice: practical, food-based, dietitian-level guidance including a 1-day sample menu, rationale, and food-safety notes.
Do NOT include Summary or Questions.

Patient report:
\"\"\"{all_lab_text}\"\"\"
User follow-up request:
\"\"\"{follow_up_q}\"\"\"
"""
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role":"user","content":prompt_ref}],
                            max_tokens=900,
                            temperature=0.2
                        )
                        ai_text = resp.choices[0].message.content
                        # try to extract Dietary Advice; fallback to whole text
                        dietary_section = extract_section(ai_text, "Dietary Advice")
                        # if model didn't use header, use entire response
                        dietary_section = dietary_section if dietary_section and dietary_section!="No findings." else ai_text.strip()
                        st.session_state['dietary'] = dietary_section
                        st.session_state['ai_raw'] = ai_text
                        st.success("Dietary advice updated.")
                    except Exception as e:
                        st.error(f"Refine (dietary) failed: {e}")

                elif refine_choice == "Suggested Questions for the Doctor":
                    prompt_ref = f"""
You are a clinical-support assistant. Respond only in English.
Provide updated Summary and Questions only. DO NOT include dietary advice.

Patient report:
\"\"\"{all_lab_text}\"\"\"

Previous AI output (for context):
\"\"\"{st.session_state.get('ai_raw','')}\"\"\"

User follow-up:
\"\"\"{follow_up_q}\"\"\"
"""
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role":"user","content":prompt_ref}],
                            max_tokens=700,
                            temperature=0.2
                        )
                        ai_text = resp.choices[0].message.content
                        raw_summary = extract_section(ai_text, "Summary")
                        raw_questions = extract_section(ai_text, "Questions")
                        cleaned_questions = clean_questions_text(raw_questions)
                        st.session_state['summary'] = raw_summary
                        st.session_state['questions'] = cleaned_questions
                        st.session_state['ai_raw'] = ai_text
                        st.success("Summary and questions updated.")
                    except Exception as e:
                        st.error(f"Refine (questions) failed: {e}")

                else:  # Both
                    prompt_ref = f"""
You are a clinical-support assistant. Respond only in English.
Provide updated Summary, Questions, and Dietary Advice (label each section with the headers: Summary, Questions, Dietary Advice).
- Summary: 2-4 short sentences.
- Questions: 3 practical questions for the doctor (do NOT include diet instructions here).
- Dietary Advice: dietitian-level, 1-day sample menu, rationale, and food-safety notes.

Patient report:
\"\"\"{all_lab_text}\"\"\"

Previous AI output (for context):
\"\"\"{st.session_state.get('ai_raw','')}\"\"\"
User follow-up:
\"\"\"{follow_up_q}\"\"\"
"""
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role":"user","content":prompt_ref}],
                            max_tokens=1000,
                            temperature=0.2
                        )
                        ai_text = resp.choices[0].message.content
                        raw_summary = extract_section(ai_text, "Summary")
                        raw_questions = extract_section(ai_text, "Questions")
                        dietary_section = extract_section(ai_text, "Dietary Advice")
                        # fallback if model didn't use header
                        if not dietary_section or dietary_section=="No findings.":
                            # try to pull Nutrition or entire response
                            dietary_section = extract_section(ai_text, "Nutrition")
                        dietary_section = dietary_section if dietary_section and dietary_section!="No findings." else ai_text.strip()
                        cleaned_questions = clean_questions_text(raw_questions)
                        st.session_state['summary'] = raw_summary
                        st.session_state['questions'] = cleaned_questions
                        st.session_state['dietary'] = dietary_section
                        st.session_state['ai_raw'] = ai_text
                        st.success("Summary, questions and dietary advice updated.")
                    except Exception as e:
                        st.error(f"Refine (both) failed: {e}")

# After refine, show current values (if generated)
if st.session_state.get('generated'):
    st.markdown("### Current Summary / Questions / Dietary Advice (most recent)")
    st.write(st.session_state.get('summary',''))
    st.write(st.session_state.get('questions',''))
    st.text_area("Dietary Advice (copyable)", value=st.session_state.get('dietary',''), height=320)
    with st.expander("Full AI output (latest)"):
        st.code(st.session_state.get('ai_raw',''))

# -------------------------
# Optional email sending
# -------------------------
st.markdown("---")
st.subheader("4) Optional: send final report via email")
send_email = st.checkbox("Send final report via email")
if send_email:
    recipient = st.text_input("Recipient email address")
    if st.button("Send Email"):
        if not recipient:
            st.error("Enter recipient email address first.")
        elif not st.session_state.get('generated'):
            st.error("Generate a report first.")
        else:
            try:
                msg = EmailMessage()
                msg['Subject'] = "Personalized Health Summary & Dietary Advice"
                sender = st.secrets.get("EMAIL_SENDER", "noreply@example.com")
                msg['From'] = sender
                msg['To'] = recipient
                body = (
                    f"Health Index: {st.session_state.get('health_index','N/A')}\n\n"
                    f"Summary:\n{st.session_state.get('summary','No findings.')}\n\n"
                    f"Questions:\n{st.session_state.get('questions','No findings.')}\n\n"
                    f"Dietary Advice:\n{st.session_state.get('dietary','No findings.')}"
                )
                msg.set_content(body)

                smtp_host = st.secrets.get("SMTP_HOST", "localhost")
                smtp_port = int(st.secrets.get("SMTP_PORT", 25))
                smtp_user = st.secrets.get("SMTP_USER")
                smtp_pass = st.secrets.get("SMTP_PASS")

                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    if smtp_user and smtp_pass:
                        server.starttls()
                        server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
                st.success(f"Report sent to {recipient}")
            except Exception as e:
                st.error(f"Failed to send email: {e}")









































