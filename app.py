# app.py
import streamlit as st
import numpy as np
from PIL import Image
import easyocr
import openai
import requests
import html
import streamlit.components.v1 as components

# ------------------ Secrets (Streamlit Secrets) ------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]
MAILGUN_API_KEY = st.secrets.get("MAILGUN_API_KEY")
MAILGUN_DOMAIN = st.secrets.get("MAILGUN_DOMAIN")
EMAIL_SENDER = f"postmaster@{MAILGUN_DOMAIN}" if MAILGUN_DOMAIN else None

# ------------------ Constants ------------------
MAX_OCR_CHARS = 3000

# ------------------ Init ------------------
st.set_page_config(page_title="AI Cancer Chatbot", layout="wide")
reader = None
try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    st.warning("EasyOCR init failed — OCR will be disabled. You can still paste text manually.")
    st.write(e)

if "conversation" not in st.session_state:
    # conversation is list of dicts: {"role":"user"/"assistant", "content": str}
    st.session_state.conversation = []

# ------------------ Helpers ------------------
def ocr_extract(uploaded_files):
    texts = []
    if not reader:
        return ""
    for f in uploaded_files:
        try:
            img = Image.open(f).convert("RGB")
            txts = reader.readtext(np.array(img), detail=0)
            texts.append(" ".join(txts))
        except Exception as e:
            st.error(f"OCR failed for {getattr(f, 'name', 'file')}: {e}")
    combined = " ".join(texts)
    return combined[:MAX_OCR_CHARS]

def call_openai_chat(messages):
    """
    messages: list of {"role": "user"|"assistant"|"system", "content": "..."}
    Uses new OpenAI python client API for chat completions.
    """
    try:
        resp = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=messages
            # no temperature for gpt-5-mini (model only supports default)
        )
        return resp.choices[0].message.content
    except Exception as e:
        # bubble error into app and return None
        st.error(f"AI generation failed: {e}")
        return None

def send_mailgun_email(to_email, subject, text):
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN):
        return {"ok": False, "status": None, "detail": "Mailgun not configured in secrets."}
    try:
        resp = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={"from": EMAIL_SENDER,
                  "to": [to_email],
                  "subject": subject,
                  "text": text},
            timeout=15
        )
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        return {"ok": resp.status_code in (200,201,202), "status": resp.status_code, "detail": detail}
    except Exception as e:
        return {"ok": False, "status": None, "detail": str(e)}

# ------------------ UI: Upload / OCR / Manual paste ------------------
st.title("AI-Driven Personalized Cancer Care Chatbot")

col1, col2 = st.columns([1, 1])
with col1:
    uploaded_files = st.file_uploader(
        "Upload medical reports (JPG/PNG/PDF). You can upload multiple files.",
        type=["jpg","jpeg","png","pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader("Uploaded files (preview)")
        for f in uploaded_files:
            try:
                if f.type.startswith("image"):
                    st.image(f, caption=f.name, use_column_width=True)
                else:
                    st.write(f.name)
            except Exception:
                st.write(f"Preview not available for {getattr(f,'name','file')}")

    manual_text = st.text_area("Or paste a short lab/test excerpt here (English, recommended):", height=160)

with col2:
    st.subheader("Controls")
    generate_clicked = st.button("Generate Summary & Dietary Advice")
    st.markdown(
        "Notes:\n"
        "- Responses are in English.\n"
        "- Dietary advice is dietitian-level and includes a 1-day sample menu.\n"
        "- Use the chat box below to refine advice; AI replies show on the left, your messages on the right."
    )

# ------------------ Handle Generate ------------------
ocr_text = ""
if uploaded_files:
    ocr_text = ocr_extract(uploaded_files)
    if ocr_text:
        st.info(f"OCR extracted (truncated to {MAX_OCR_CHARS} chars).")

if generate_clicked:
    prompt_text = manual_text.strip() or ocr_text.strip()
    if not prompt_text:
        st.warning("Please upload a report or paste text before generating.")
    else:
        # Build messages with system instruction and history context (we'll only send system + user)
        messages = [{"role":"system","content":"You are an expert oncology dietitian and clinical-support assistant. Provide concise clinical summary, 3 suggested practical questions for next doctor visit (not diet), and dietitian-level dietary advice (include clear 1-day sample menu separated by spacing). Write in plain English."},
                    {"role":"user","content": prompt_text}]
        ai_text = call_openai_chat(messages)
        if ai_text:
            st.session_state.conversation.append({"role":"assistant","content": ai_text})
            # After generation, render chat below (auto-scroll handled in component HTML)

# ------------------ Chat input (bottom) ------------------
st.markdown("---")
st.subheader("Chat — refine advice or ask questions")
chat_input = st.text_input("Type your follow-up or refinement here (press Send):", "")

send_clicked = st.button("Send")
if send_clicked and chat_input.strip():
    # append user's message
    st.session_state.conversation.append({"role":"user","content": chat_input.strip()})
    # prepare full conversation as messages for context: we include a system message and then the conversation
    messages = [{"role":"system","content":"You are an expert oncology dietitian and clinical-support assistant. Provide concise clinical summary, 3 suggested practical questions for next doctor visit (not diet), and dietitian-level dietary advice (include clear 1-day sample menu separated by spacing). Write in plain English."}]
    # Convert st.session_state.conversation to API messages (role mapping)
    for turn in st.session_state.conversation:
        # API expects 'user' and 'assistant' roles; map accordingly
        if turn["role"] == "user":
            messages.append({"role":"user","content": turn["content"]})
        else:
            messages.append({"role":"assistant","content": turn["content"]})
    ai_reply = call_openai_chat(messages)
    if ai_reply:
        st.session_state.conversation.append({"role":"assistant","content": ai_reply})

# ------------------ Render scrollable chat with left/right bubbles and copy buttons ------------------
st.markdown("### Conversation (latest at bottom)")

# Build HTML for the chat container, messages, copy buttons and auto-scroll to bottom
def build_chat_html(conversation):
    # Escape content for safe embedding
    def esc(s): return html.escape(s).replace("\n","<br>")
    html_msgs = []
    for idx, m in enumerate(conversation):
        role = m["role"]
        content = esc(m["content"])
        if role == "assistant":
            # left bubble
            bubble = f"""
            <div class="msg-row left">
              <div class="bubble">
                <div class="bubble-copy">
                  <button class="copy-btn" data-idx="{idx}">Copy</button>
                </div>
                <div class="bubble-content">{content}</div>
              </div>
            </div>
            """
        else:
            # right bubble
            bubble = f"""
            <div class="msg-row right">
              <div class="bubble user">
                <div class="bubble-copy">
                  <button class="copy-btn" data-idx="{idx}">Copy</button>
                </div>
                <div class="bubble-content">{content}</div>
              </div>
            </div>
            """
        html_msgs.append(bubble)
    # full container with styles and JS for copy + autoscroll
    html_doc = f"""
    <html>
    <head>
    <meta charset="utf-8"/>
    <style>
      :root{{font-family: Arial, sans-serif;}}
      .chat-wrap {{
        width: 800px;
        max-height: 520px;
        border: 1px solid #ddd;
        padding: 16px;
        overflow-y: auto;
        background: #fff;
        margin: 0 auto;
      }}
      .msg-row {{ display: flex; margin: 8px 0; }}
      .msg-row.left {{ justify-content: flex-start; }}
      .msg-row.right {{ justify-content: flex-end; }}
      .bubble {{
        max-width: 78%;
        background: #f1f3f5;
        padding: 10px 12px;
        border-radius: 12px;
        position: relative;
      }}
      .bubble.user {{ background: #d1e7dd; }}
      .bubble-copy {{ position: absolute; right: 8px; top: 6px; }}
      .copy-btn {{
        font-size:12px;
        padding:4px 8px;
        border-radius:6px;
        border:1px solid #aaa;
        background:#fff;
        cursor:pointer;
      }}
      .bubble-content {{ white-space: pre-wrap; word-wrap: break-word; }}
    </style>
    </head>
    <body>
      <div id="chat" class="chat-wrap">
        {''.join(html_msgs)}
      </div>

      <script>
        // copy button handler
        const btns = document.querySelectorAll('.copy-btn');
        btns.forEach(b => {{
          b.addEventListener('click', async (e) => {{
            const idx = b.getAttribute('data-idx');
            const bubbles = document.querySelectorAll('.bubble-content');
            const text = bubbles[idx].innerText;
            try {{
              await navigator.clipboard.writeText(text);
              b.innerText = 'Copied';
              setTimeout(()=> b.innerText = 'Copy', 1200);
            }} catch(err) {{
              console.error('Clipboard error', err);
              b.innerText = 'Failed';
              setTimeout(()=> b.innerText = 'Copy', 1200);
            }}
          }});
        }});
        // autoscroll to bottom
        const chat = document.getElementById('chat');
        chat.scrollTop = chat.scrollHeight;
      </script>
    </body>
    </html>
    """
    return html_doc

chat_html = build_chat_html(st.session_state.conversation)
components.html(chat_html, height=560)

# ------------------ Email sending with improved error messages ------------------
st.markdown("---")
st.subheader("Send final report via Mailgun (optional)")

email_to = st.text_input("Recipient email (for Mailgun sandbox: recipient must be authorised)")

if st.button("Send Report"):
    if not email_to:
        st.warning("Please enter a recipient email.")
    elif not st.session_state.conversation:
        st.warning("No conversation to send. Generate summary first.")
    else:
        final_text = "\n\n".join(
            [f"{('You' if m['role']=='user' else 'AI')}: {m['content']}" for m in st.session_state.conversation]
        )
        result = send_mailgun_email(email_to, "AI Health Summary & Dietary Advice", final_text)
        if result["ok"]:
            st.success("Email sent successfully.")
        else:
            # Show helpful guidance for 403 (Forbidden) common case
            if result["status"] == 403:
                st.error("Mailgun returned 403 Forbidden. Likely causes:")
                st.markdown(
                    "- You are using a Mailgun **sandbox** domain and the recipient address is not authorised. "
                    "Go to your Mailgun dashboard → Sending → Domains → click the sandbox domain → 'Authorized Recipients' and add the recipient.\n\n"
                    "- Or the MAILGUN_API_KEY or MAILGUN_DOMAIN in Streamlit Secrets is incorrect.\n\n"
                    "What to do:\n"
                    "1) If using sandbox domain: add and verify the recipient email in Mailgun's sandbox settings; try again.\n"
                    "2) If using a custom domain: ensure DNS records (TXT/MX) are verified in Mailgun and domain status is active.\n"
                    "3) Double-check the MAILGUN_API_KEY and MAILGUN_DOMAIN values in Streamlit Secrets."
                )
            else:
                st.error("Failed to send email. Debug info:")
            st.write(result)























































