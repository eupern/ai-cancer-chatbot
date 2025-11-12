# app.py (top portion fixed)

import streamlit as st
import os
import re
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)  # English-first OCR

# Streamlit config
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("AI-Driven Personalized Cancer Care Chatbot")

# Short, single-line instruction
st.write("Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get an English health summary, doctor questions, and nutrition advice.")

# OpenAI API setup
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found in Streamlit Secrets. Paste below for this session:")
    api_key_input = st.text_input("OpenAI API Key (session only):", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# UI: file upload and text input
st.subheader("1) Input medical reports or lab summary")
uploaded_files = st.file_uploader(
    "Upload medical reports / imaging files (JPG/PNG/PDF). You can upload multiple files.",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)
text_input = st.text_area("Or paste a short lab/test excerpt here (English preferred)", height=160)


































