# ai-cancer-chatbot
AI-Driven Personalized Cancer Care Chatbot

## 🏥 Problem Statement
Cancer patients struggle to understand complex medical reports and need personalized dietary guidance between doctor visits.

## 🚀 Solution
AI-powered web app that:
- Extracts text from medical documents using OCR
- Generates health summaries using GPT-4o
- Provides personalized dietary advice
- Allows interactive Q&A about medical reports
- Sends conversation summaries via email

## 🛠 Tech Stack
- **Frontend**: Streamlit
- **AI/ML**: OpenAI gpt-5-mini, EasyOCR
- **Email**: Gmail SMTP
- **Deployment**: Streamlit Cloud

## 📦 Installation
```bash
pip install -r requirements.txt
streamlit run app.py
