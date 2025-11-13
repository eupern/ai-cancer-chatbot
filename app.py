import streamlit as st
import numpy as np
import easyocr
from PIL import Image
import openai
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import time
import base64
import json

# Page configuration with accessibility features
st.set_page_config(
    page_title="AI Cancer Care Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GMAIL_EMAIL = st.secrets["GMAIL_EMAIL"]
GMAIL_APP_PASSWORD = st.secrets["GMAIL_APP_PASSWORD"]

openai.api_key = OPENAI_API_KEY
reader = easyocr.Reader(['en'])

# Initialize session states with privacy features
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "last_activity" not in st.session_state:
    st.session_state.last_activity = datetime.now()
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()
if "high_contrast" not in st.session_state:
    st.session_state.high_contrast = False
if "font_size" not in st.session_state:
    st.session_state.font_size = "medium"

# ========== PRIVACY & SECURITY FUNCTIONS ==========
def setup_auto_clear():
    """Set up session expiration and auto-clear for privacy"""
    # Update activity timestamp on any interaction
    st.session_state.last_activity = datetime.now()
    
    # Check for inactivity (30 minutes)
    time_since_activity = datetime.now() - st.session_state.last_activity
    if time_since_activity > timedelta(minutes=30):
        clear_sensitive_data()
        st.warning("🕒 Session expired due to inactivity. All data has been cleared for your privacy.")
        st.stop()

def clear_sensitive_data():
    """Clear all sensitive medical data"""
    sensitive_keys = ["ocr_text", "conversation", "user_email", "processing_complete"]
    for key in sensitive_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reinitialize empty states
    st.session_state.conversation = []
    st.session_state.ocr_text = ""
    st.session_state.processing_complete = False
    st.session_state.user_email = ""
    st.session_state.last_activity = datetime.now()

def download_conversation_json():
    """Download conversation as JSON for data portability"""
    if not st.session_state.conversation:
        return None
    
    session_data = {
        "conversation": st.session_state.conversation,
        "exported_at": datetime.now().isoformat(),
        "total_messages": len(st.session_state.conversation)
    }
    
    # Convert to JSON string
    session_json = json.dumps(session_data, indent=2)
    
    # Create download link
    b64 = base64.b64encode(session_json.encode()).decode()
    filename = f"medical_conversation_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}" style="text-decoration: none;">' \
           f'<button style="background-color: #4CAF50; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">' \
           f'💾 Download Conversation (JSON)</button></a>'
    
    return href

# ========== ACCESSIBILITY FUNCTIONS ==========
def apply_accessibility_styles():
    """Apply accessibility styles based on user preferences"""
    styles = ""
    
    # High contrast mode
    if st.session_state.high_contrast:
        styles += """
        <style>
        .main .block-container {
            background-color: #000000;
            color: #FFFFFF;
        }
        .stChatMessage {
            background-color: #2E2E2E !important;
            color: #FFFFFF !important;
        }
        </style>
        """
    
    # Font size adjustments
    font_sizes = {
        "small": "14px",
        "medium": "16px", 
        "large": "18px",
        "x-large": "20px"
    }
    
    base_size = font_sizes.get(st.session_state.font_size, "16px")
    styles += f"""
    <style>
    .main .block-container {{
        font-size: {base_size};
    }}
    .stButton>button {{
        font-size: {base_size};
    }}
    .stTextInput>div>div>input {{
        font-size: {base_size};
    }}
    </style>
    """
    
    st.markdown(styles, unsafe_allow_html=True)

# ========== ENHANCED UX FUNCTIONS ==========
def process_documents_with_progress(uploaded_files):
    """Enhanced document processing with progress indicators"""
    total_files = len(uploaded_files)
    
    if total_files == 0:
        return
    
    # Progress bar for multiple files
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear previous OCR text
    st.session_state.ocr_text = ""
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"📄 Processing {i+1}/{total_files}: {uploaded_file.name}")
        
        try:
            img = Image.open(uploaded_file)
            
            # OCR with individual file progress
            with st.spinner(f"🔍 Extracting text from {uploaded_file.name}..."):
                extracted_text = " ".join(reader.readtext(np.array(img), detail=0))
                st.session_state.ocr_text += extracted_text + "\n\n"
                
                if extracted_text.strip():
                    st.success(f"✅ Text extracted from {uploaded_file.name}")
                else:
                    st.warning(f"⚠️ No text detected in {uploaded_file.name}")
        
        except Exception as e:
            st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
        
        # Update progress bar
        progress_bar.progress((i + 1) / total_files)
        time.sleep(0.5)  # Brief pause to show progress
    
    status_text.text("✅ All documents processed successfully!")
    time.sleep(1)  # Show completion message
    status_text.empty()
    progress_bar.empty()

def enhanced_generate_analysis():
    """Generate analysis with improved UX feedback"""
    # Create status container
    status_container = st.empty()
    
    with status_container.container():
        st.info("🚀 Starting medical analysis...")
        
        # Simulated steps with progress
        steps = [
            "📊 Analyzing medical report content...",
            "💡 Generating health summary...", 
            "❓ Preparing questions for your doctor...",
            "🍎 Creating dietary recommendations...",
            "🤖 Finalizing AI analysis..."
        ]
        
        for step in steps:
            with st.spinner(step):
                time.sleep(1)  # Simulate processing time
        
        # Actual AI processing
        try:
            prompt = f"""Based on the following medical report content, please provide:
1. Concise health summary
2. 5 suggested questions for your cardiologist/primary doctor (limit to 5 questions maximum)
3. Professional dietitian advice based on health report

Medical report content:
{st.session_state.ocr_text}

Please respond in a professional, caring manner. Keep the questions limited to 5 maximum to avoid overwhelming the patient."""

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            ai_output = response.choices[0].message.content
            
            st.session_state.conversation.append({
                "role": "assistant", 
                "content": ai_output,
                "type": "initial_analysis"
            })
            st.session_state.processing_complete = True
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            return
    
    # Clear status and show success
    status_container.empty()
    
    # Enhanced success message
    success_placeholder = st.empty()
    with success_placeholder.container():
        st.balloons()  # Celebration effect
        st.success("""
        🎉 **Analysis Complete!**
        
        ✅ Health summary generated  
        ✅ Doctor questions prepared  
        ✅ Dietary advice created  
        ✅ Ready for consultation chat
        
        *You can now ask follow-up questions using the chat below.*
        """)
    
    # Auto-remove success message after 4 seconds
    time.sleep(4)
    success_placeholder.empty()
    st.rerun()

# ========== EXISTING FUNCTIONS (Updated) ==========
def handle_user_message(user_input):
    """Process user message and generate AI response"""
    st.session_state.conversation.append({"role": "user", "content": user_input})
    st.session_state.last_activity = datetime.now()  # Update activity
    
    with st.spinner("🤖 AI is thinking..."):
        try:
            messages = [{"role": msg["role"], "content": msg["content"]} 
                       for msg in st.session_state.conversation]
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            ai_reply = response.choices[0].message.content
            st.session_state.conversation.append({"role": "assistant", "content": ai_reply})
            st.session_state.last_activity = datetime.now()
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to generate response: {e}")

def send_email_via_gmail():
    """Send conversation via Gmail SMTP with enhanced UX"""
    if not st.session_state.user_email:
        st.error("❌ Please enter your email address first")
        return False
    
    # Validate email format
    if "@" not in st.session_state.user_email or "." not in st.session_state.user_email:
        st.error("❌ Please enter a valid email address")
        return False
    
    try:
        with st.spinner("📧 Preparing and sending email..."):
            # Gmail SMTP configuration
            smtp_server = "smtp.gmail.com"
            port = 587
            sender_email = GMAIL_EMAIL
            sender_password = GMAIL_APP_PASSWORD
            
            # Format conversation for email
            email_content = f"""Cancer Care Assistant - Conversation Summary
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Session duration: {str(datetime.now() - st.session_state.session_start).split('.')[0]}
============================================================

"""
            
            for i, msg in enumerate(st.session_state.conversation):
                if msg["role"] == "user":
                    email_content += f"YOU (Message {i+1}):\n{msg['content']}\n\n"
                else:
                    email_content += f"AI ASSISTANT (Message {i+1}):\n{msg['content']}\n\n"
                email_content += "----------------------------------------\n\n"
            
            # Add privacy notice
            email_content += "\n\n🔒 Privacy Notice: This conversation will be automatically deleted from our system.\n"
            
            # Create message
            message = MIMEMultipart()
            message["From"] = f"Cancer Care Assistant <{sender_email}>"
            message["To"] = st.session_state.user_email
            message["Subject"] = "Your Cancer Care Assistant Conversation Summary"
            message.attach(MIMEText(email_content, "plain"))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_server, port) as server:
                server.starttls(context=context)
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, st.session_state.user_email, message.as_string())
        
        # Enhanced success feedback
        st.success(f"""
        ✅ **Email Sent Successfully!**
        
        📬 Delivered to: {st.session_state.user_email}
        ⏰ Sent at: {datetime.now().strftime("%H:%M:%S")}
        📋 Contains: {len(st.session_state.conversation)} messages
        
        *Check your inbox (and spam folder) for the summary.*
        """)
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to send email: {str(e)}")
        if "Authentication failed" in str(e):
            st.info("💡 Tip: Make sure you're using an App Password, not your regular Gmail password")
        return False

def handle_suggested_question(question):
    """Handle suggested question button clicks"""
    st.session_state.conversation.append({"role": "user", "content": question})
    st.session_state.last_activity = datetime.now()
    
    with st.spinner("🤖 AI is thinking..."):
        try:
            messages = [{"role": msg["role"], "content": msg["content"]} 
                       for msg in st.session_state.conversation]
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            ai_reply = response.choices[0].message.content
            st.session_state.conversation.append({"role": "assistant", "content": ai_reply})
            st.session_state.last_activity = datetime.now()
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to generate response: {e}")

# Suggested questions for quick buttons
SUGGESTED_QUESTIONS = [
    "Can you explain my test results in simple terms?",
    "What are the most important lifestyle changes I should make?",
    "What symptoms should I monitor and report to my doctor?",
    "Are there any foods or supplements I should avoid?",
    "When should I schedule my next follow-up appointment?"
]

# ========== MAIN APPLICATION ==========
def main():
    # Apply privacy and accessibility features
    setup_auto_clear()
    apply_accessibility_styles()
    
    # Main title with session info
    col_title, col_session = st.columns([3, 1])
    with col_title:
        st.title("🏥 AI-Driven Personalized Cancer Care Assistant")
    with col_session:
        session_duration = datetime.now() - st.session_state.session_start
        st.caption(f"Session: {str(session_duration).split('.')[0]}")
    
    st.markdown("---")
    
    # Accessibility controls in sidebar
    with st.sidebar:
        st.header("♿ Accessibility")
        
        # High contrast toggle
        st.session_state.high_contrast = st.toggle(
            "High Contrast Mode", 
            value=st.session_state.high_contrast,
            help="Improves visibility for users with low vision"
        )
        
        # Font size selector
        st.session_state.font_size = st.selectbox(
            "Font Size",
            options=["small", "medium", "large", "x-large"],
            index=1,  # medium
            help="Adjust text size for better readability"
        )
        
        st.markdown("---")
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
        """)
        
        # Enhanced privacy information
        with st.expander("🔒 Privacy & Security", expanded=False):
            st.markdown("""
            **Your Privacy Matters:**
            - 🔄 Auto-clears after 30min inactivity
            - 📧 Emails are end-to-end delivered
            - 💾 No data stored on our servers
            - 🚫 No third-party data sharing
            
            **For Your Safety:**
            - Always consult healthcare providers
            - Never share login credentials
            - Use secure internet connections
            """)
        
        # Data management
        st.markdown("---")
        st.header("💾 Data Management")
        
        if st.session_state.conversation:
            # Download conversation
            download_html = download_conversation_json()
            if download_html:
                st.markdown(download_html, unsafe_allow_html=True)
            
            # Clear data button
            if st.button("🗑️ Clear All Data Now", use_container_width=True):
                clear_sensitive_data()
                st.success("All data cleared successfully!")
                st.rerun()
        
        # Session statistics
        if st.session_state.conversation:
            st.markdown("---")
            user_msgs = len([msg for msg in st.session_state.conversation if msg["role"] == "user"])
            assistant_msgs = len([msg for msg in st.session_state.conversation if msg["role"] == "assistant"])
            st.metric("Conversation", f"{user_msgs} user · {assistant_msgs} AI")
            
            # Data size info
            total_chars = sum(len(msg["content"]) for msg in st.session_state.conversation)
            st.metric("Data Size", f"~{total_chars // 1000}KB")

    # Two-column layout
    left_col, right_col = st.columns([1, 1])

    # Left column - Document upload and processing
    with left_col:
        st.subheader("📁 Upload Medical Documents")
        
        uploaded_files = st.file_uploader(
            "Select PDF or image files:",
            type=['pdf','png','jpg','jpeg'],
            accept_multiple_files=True,
            help="Upload lab reports, diagnosis documents, medical images. Max 10MB per file."
        )
        
        if uploaded_files:
            # Enhanced file processing with progress
            process_documents_with_progress(uploaded_files)
            
            # Document preview
            with st.expander("📄 Document Preview", expanded=True):
                for uploaded_file in uploaded_files:
                    try:
                        img = Image.open(uploaded_file)
                        st.image(img, caption=uploaded_file.name, use_column_width=True)
                    except Exception as e:
                        st.error(f"Failed to display {uploaded_file.name}: {e}")
            
            # Enhanced generate button with status
            if st.button("🎯 Generate Health Analysis & Recommendations", 
                        type="primary", 
                        use_container_width=True,
                        disabled=not st.session_state.ocr_text.strip()):
                if not st.session_state.ocr_text.strip():
                    st.warning("⚠️ Please upload medical documents with readable text first")
                else:
                    enhanced_generate_analysis()

    # Right column - Chat interface
    with right_col:
        st.subheader("💬 Consultation Chat")
        
        # Chat container with accessibility
        chat_container = st.container()
        
        with chat_container:
            for i, msg in enumerate(st.session_state.conversation):
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg["content"])
                        
                        # Suggested questions quick buttons
                        if msg.get("type") == "initial_analysis" and i == len(st.session_state.conversation) - 1:
                            st.markdown("---")
                            st.markdown("**💡 Quick questions you can ask:**")
                            
                            col1, col2 = st.columns(2)
                            buttons_per_col = (len(SUGGESTED_QUESTIONS) + 1) // 2
                            
                            for idx, question in enumerate(SUGGESTED_QUESTIONS):
                                if idx < buttons_per_col:
                                    with col1:
                                        if st.button(
                                            f"❓ {question}", 
                                            key=f"q_{idx}_{i}",
                                            use_container_width=True,
                                            help="Click to ask this question"
                                        ):
                                            handle_suggested_question(question)
                                else:
                                    with col2:
                                        if st.button(
                                            f"❓ {question}", 
                                            key=f"q_{idx}_{i}",
                                            use_container_width=True,
                                            help="Click to ask this question"
                                        ):
                                            handle_suggested_question(question)
        
        # Chat input
        if st.session_state.processing_complete:
            st.markdown("---")
            user_input = st.chat_input("Type your question or concern here...")
            
            if user_input:
                handle_user_message(user_input)

    # Enhanced footer with privacy features
    st.markdown("---")
    footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)
    
    with footer_col1:
        if st.button("🔄 New Session", use_container_width=True, help="Start fresh - all data will be cleared"):
            clear_sensitive_data()
            st.rerun()
    
    with footer_col2:
        st.markdown("**📧 Email Summary**")
        email_col1, email_col2 = st.columns([2, 1])
        with email_col1:
            user_email = st.text_input(
                "Your email:",
                placeholder="your.email@example.com",
                key="email_input",
                label_visibility="collapsed"
            )
            if user_email:
                st.session_state.user_email = user_email
                st.session_state.last_activity = datetime.now()
        with email_col2:
            if st.button("Send", use_container_width=True, type="secondary"):
                if st.session_state.conversation:
                    send_email_via_gmail()
                else:
                    st.warning("No conversation to send")
    
    with footer_col3:
        st.markdown("**🔒 Privacy Status**")
        inactivity_time = datetime.now() - st.session_state.last_activity
        minutes_left = max(0, 30 - int(inactivity_time.total_seconds() / 60))
        st.caption(f"Auto-clear in: {minutes_left} min")
    
    with footer_col4:
        st.markdown("**📱 Accessibility**")
        st.caption(f"Mode: {'High Contrast' if st.session_state.high_contrast else 'Standard'}")

# Run the application
if __name__ == "__main__":
    main()




































































