import streamlit as st

st.title("AI-Driven Personalized Cancer Care Chatbot")
st.write("Welcome! This app provides personalized cancer care insights, nutritional suggestions, and questions for doctor consultations.")

user_input = st.text_input("Enter your health concern or recent test result:")
if user_input:
    st.write("✅ Processing your input...")
    st.write("🤖 AI suggestion: Please consult your doctor for professional medical advice.")

