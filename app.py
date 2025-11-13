import streamlit as st
import numpy as np
import easyocr
from PIL import Image
import openai
import requests

# Page configuration
st.set_page_config(
    page_title="AI-Driven Personalized Cancer Care Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MAILGUN_API_KEY = st.secrets["MAILGUN_API_KEY"]
MAILGUN_DOMAIN = st.secrets["MAILGUN_DOMAIN"]
EMAIL_SENDER = f"postmaster@{MAILGUN_DOMAIN}"

openai.api_key = OPENAI_API_KEY
reader = easyocr.Reader(['en'])

# Initialize session states
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# Functions
def generate_analysis():
    """Generate initial health analysis and dietary advice"""
    with st.spinner("🤖 AI is analyzing your health condition and generating personalized recommendations..."):
        try:
            prompt = f"""Based on the following medical report content, please provide:
1. Concise health summary
2. 5 suggested questions for your cardiologist/primary doctor (limit to 5 questions maximum)
3. Professional dietitian advice based on health report

Medical report content:
{st.session_state.ocr_text}

Please respond in a professional, caring manner. Keep the questions limited to 5 maximum to avoid overwhelming the patient."""

            response = openai.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for better performance
                messages=[{"role": "user", "content": prompt}]
            )
            ai_output = response.choices[0].message.content
            
            st.session_state.conversation.append({
                "role": "assistant", 
                "content": ai_output,
                "type": "initial_analysis"
            })
            st.session_state.processing_complete = True
            st.success("✅ Analysis completed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")

def handle_user_message(user_input):
    """Process user message and generate AI response"""
    st.session_state.conversation.append({"role": "user", "content": user_input})
    
    with st.spinner("🤖 AI is thinking..."):
        try:
            messages = [{"role": msg["role"], "content": msg["content"]} 
                       for msg in st.session_state.conversation]
            
            response = openai.chat.completions.create(
                model="gpt-5-mini",
                messages=messages
            )
            ai_reply = response.choices[0].message.content
            st.session_state.conversation.append({"role": "assistant", "content": ai_reply})
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to generate response: {e}")

def send_email_via_mailgun():
    """Send conversation via Mailgun email"""
    if not st.session_state.user_email:
        st.error("❌ Please enter your email address first")
        return False
    
    try:
        # Format conversation for email
        email_content = "Cancer Care Assistant Conversation Summary\n\n"
        email_content += "=" * 50 + "\n\n"
        
        for msg in st.session_state.conversation:
            if msg["role"] == "user":
                email_content += f"YOU: {msg['content']}\n\n"
            else:
                email_content += f"AI ASSISTANT: {msg['content']}\n\n"
            email_content += "-" * 30 + "\n\n"
        
        # Mailgun API request
        response = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": f"Cancer Care Assistant <{EMAIL_SENDER}>",
                "to": [st.session_state.user_email],
                "subject": "Your Cancer Care Assistant Conversation Summary",
                "text": email_content
            }
        )
        
        if response.status_code == 200:
            st.success(f"✅ Conversation sent to {st.session_state.user_email} successfully!")
            return True
        else:
            st.error(f"❌ Failed to send email: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        st.error(f"❌ Email sending failed: {e}")
        return False

# Suggested questions for quick buttons
SUGGESTED_QUESTIONS = [
    "Can you explain my test results in simple terms?",
    "What are the most important lifestyle changes I should make?",
    "What symptoms should I monitor and report to my doctor?",
    "Are there any foods or supplements I should avoid?",
    "When should I schedule my next follow-up appointment?"
]

# Main title
st.title("🏥 AI-Driven Personalized Cancer Care Assistant")
st.markdown("---")

# Two-column layout
left_col, right_col = st.columns([1, 1])

# Left column - Document upload and processing
with left_col:
    st.subheader("📁 Upload Medical Documents")
    
    uploaded_files = st.file_uploader(
        "Select PDF or image files:",
        type=['pdf','png','jpg','jpeg'],
        accept_multiple_files=True,
        help="Upload lab reports, diagnosis documents, medical images"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded successfully")
        
        # Document preview with expander
        with st.expander("📄 Document Preview", expanded=True):
            for uploaded_file in uploaded_files:
                try:
                    img = Image.open(uploaded_file)
                    st.image(img, caption=uploaded_file.name, use_column_width=True)
                    
                    # OCR processing with progress
                    with st.spinner(f"Analyzing {uploaded_file.name}..."):
                        st.session_state.ocr_text += " ".join(reader.readtext(np.array(img), detail=0)) + "\n\n"
                        
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
        
        # Primary action button
        if st.button("🎯 Generate Health Analysis & Recommendations", 
                    type="primary", 
                    use_container_width=True):
            if not st.session_state.ocr_text.strip():
                st.warning("⚠️ Please upload medical documents first")
            else:
                generate_analysis()

# Right column - Chat interface
with right_col:
    st.subheader("💬 Consultation Chat")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for i, msg in enumerate(st.session_state.conversation):
            if msg["role"] == "user":
                # User message styling
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content"])
            else:
                # AI message styling
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg["content"])
                    
                    # Suggested questions quick buttons
                    if msg.get("type") == "initial_analysis":
                        st.markdown("---")
                        st.markdown("**💡 Suggested questions you might want to ask:**")
                        
                        # Create 2 columns for better button layout
                        col1, col2 = st.columns(2)
                        
                        buttons_per_col = (len(SUGGESTED_QUESTIONS) + 1) // 2
                        
                        for idx, question in enumerate(SUGGESTED_QUESTIONS):
                            if idx < buttons_per_col:
                                with col1:
                                    if st.button(f"❓ {question}", key=f"q_{idx}", use_container_width=True):
                                        st.session_state.conversation.append({
                                            "role": "user", 
                                            "content": question
                                        })
                                        st.rerun()
                            else:
                                with col2:
                                    if st.button(f"❓ {question}", key=f"q_{idx}", use_container_width=True):
                                        st.session_state.conversation.append({
                                            "role": "user", 
                                            "content": question
                                        })
                                        st.rerun()
    
    # Chat input - only show after initial analysis
    if st.session_state.processing_complete:
        st.markdown("---")
        user_input = st.chat_input("Type your question or concern here...")
        
        if user_input:
            handle_user_message(user_input)

# Bottom toolbar
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    if st.button("🔄 Start New Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
with footer_col2:
    # Email input and send button
    email_col1, email_col2 = st.columns([2, 1])
    with email_col1:
        st.session_state.user_email = st.text_input(
            "Email for summary:",
            placeholder="your.email@example.com",
            key="email_input"
        )
    with email_col2:
        if st.button("📧 Send Summary", use_container_width=True):
            send_email_via_mailgun()
with footer_col3:
    st.markdown("🔒 **Privacy Protected** | Your data is only used for this session")

# Sidebar with instructions
with st.sidebar:
    st.header("ℹ️ User Guide")
    st.markdown("""
    1. **Upload** your medical documents
    2. **Click Generate** for initial analysis
    3. **Chat** to ask specific questions
    
    ### 📋 Supported Documents
    - Lab test reports
    - Diagnosis certificates
    - Medical imaging reports
    - Prescription documents
    
    ### ⚠️ Important Notes
    - Recommendations are for reference only
    - Always follow doctor's advice
    - Protect your privacy information
    """)
    
    # Display conversation stats
    if st.session_state.conversation:
        st.markdown("---")
        user_msgs = len([msg for msg in st.session_state.conversation if msg["role"] == "user"])
        assistant_msgs = len([msg for msg in st.session_state.conversation if msg["role"] == "assistant"])
        st.metric("Conversation Length", f"{user_msgs} user · {assistant_msgs} AI")




































































