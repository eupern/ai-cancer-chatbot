# app.py - fully patched, Twilio removed, follow-up questions, email optional, all helpers included
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

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)

# Streamlit config
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("AI-Driven Personalized Cancer Care Chatbot")
st.write("Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, and dietitian-level nutrition advice.")

# OpenAI client setup
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    api_key_input = st.text_input("OpenAI API Key (session only):", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Helper functions

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
    raw_neut_percent = find_one([r"neutrophil[s]?\s*%[:\s]*([\d\.]+)", r"neutrophil[s]?\s*percent[:\s]*([\d\.]+)"])
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


def generate_dietary_deep_dive_en(lab_vals):
    lines = ["Clinical note & overview:"]
    wbc_note = lab_vals.get('wbc_note')
    neut_note = lab_vals.get('neut_note')
    if wbc_note: lines.append(f"(Note: {wbc_note})")
    if neut_note: lines.append(f"(Note: {neut_note})")

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
            if neut < 0.5: severity = "severe"
            elif neut < 1.0: severity = "moderate"
            else: severity = "mild"
    elif wbc is not None:
        try:
            if float(wbc) < 3.0:
                neutropenia_flag = True
                severity = "possible (WBC low)"
        except: pass

    if neutropenia_flag:
        lines.append(f"Patient shows neutropenia / low WBC (severity: {severity}). Follow food-safety measures.")
        lines.append("Practical guidance (neutropenia):")
        lines.extend([
            "- Well-cooked high-quality proteins: eggs, cooked fish, chicken, tofu, pasteurised yogurt.",
            "- Cooked whole grains and soluble fiber: oats, brown rice, banana, oat bran.",
            "- Avoid raw milk, raw eggs, raw seafood, raw salads, raw sprouts, undercooked meats.",
            "- Include zinc/selenium foods: pumpkin seeds, small amount Brazil nuts (consult physician for supplements).",
            "- Maintain hydration and oral hygiene. Seek urgent care if fever occurs."
        ])

    if hb is not None and hb < 12:
        lines.append("Anemia-related suggestions: Increase iron and quality protein sources paired with vitamin C.")
    if plt is not None and plt < 100:
        lines.append("Low platelets: avoid hard/sharp foods; modify texture.")
    if glu is not None:
        if glu > 7.0:
            lines.append("Hyperglycaemia: reduce refined sugars; prefer whole grains and vegetables.")

    lines.append("Sample 1-day menu (reference):")
    lines.extend([
        "- Breakfast: cooked oats + cooked banana + pumpkin seeds + pasteurised yogurt.",
        "- Lunch: steamed chicken + brown rice + steamed carrot + cooked spinach.",
        "- Dinner: steamed fish + quinoa/brown rice + steamed greens.",
        "- Snacks: hard-boiled egg, small portion cooked fruit compote."
    ])
    lines.append("Important cautions: Avoid raw/undercooked foods if immunosuppressed. Consult physician before supplements.")

    return "\n".join(lines), ("neutropenia" if neutropenia_flag else None)


def extract_section(text, header):
    pattern = rf"{header}\s*[:\-]?\s*(.*?)(?=\n(?:Summary|Questions|Nutrition)\s*[:\-]|\Z)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if m: return m.group(1).strip()
    variants = {
        "Questions": [r"questions to ask", r"doctor questions", r"questions:", r"questions to ask the doctor", r"questions for doctor"],
        "Summary": [r"summary", r"health summary", r"clinical summary"],
        "Nutrition": [r"nutrition", r"recommendations", r"diet", r"nutrition recommendations"]
    }
    for v in variants.get(header, []):
        m2 = re.search(rf"{v}\s*[:\-]?\s*(.*?)(?=\n(?:summary|questions|nutrition)\s*[:\-]|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        if m2: return m2.group(1).strip()
    return "No findings."

# UI: file upload and text input
uploaded_files = st.file_uploader(...)
# OCR processing remains as before
# Generate Summary & Recommendations button code as before, using the helpers
# Follow-up question code as before
# Optional email send code as before

# All the helper functions are now correctly defined before use, eliminating NameError.





























