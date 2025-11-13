# app.py
import streamlit as st
import numpy as np
from PIL import Image
import easyocr
import openai
import requests
import html
import streamlit.components.v1 as components
import io
import datetime

# Optional PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ------------------ Secrets ------------------
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
    try:
        resp = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=messages
            # gpt-5-mini uses default temperature only
        )
        return resp.choices[0].message.content
    except Exception as e:
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

def authorize_mailgun_sandbox_recipient(email):
    """
    Mailgun sandbox helper:
    POST https://api.mailgun.net/v5/sandbox/auth_recipients?email=someone@example.com
    Returns dict with ok/status/detail
    """
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN):
        return {"ok": False, "status": None, "detail": "Mailgun not configured in secrets."}
    try:
        url = f"https://api.mailgun.net/v5/sandbox/auth_recipients?email={email}"
        resp = requests.post(url, auth=("api", MAILGUN_API_KEY), timeout=10)
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        return {"ok": resp.status_code in (200,201,202), "status": resp.status_code, "detail": detail}
    except Exception as e:
        return {"ok": False, "status": None, "detail": str(e)}

def build_chat_html(conversation):
    def esc(s): return html.escape(s).replace("\n","<br>")
    html_msgs = []
    for idx, m in enumerate(conversation):
        role = m["role"]
        content = esc(m["content"])
        if role == "assistant":
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
            bubble = f"""
            <div class="msg-row right">
              <div class="bubble user">
                <div class="bubble-copy">
                  <button class="copy-btn user-btn" data-idx="{idx}">Copy</button>
                </div>
                <div class="bubble-content">{content}</div>
              </div>
            </div>
            """
        html_msgs.append(bubble)

    html_doc = f"""
    <html><head><meta charset="utf-8"/>
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
        padding: 12px 14px;
        border-radius: 12px;
        position: relative;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      }}
      .bubble.user {{ background: #d1e7dd; }}
      .bubble-copy {{ position: absolute; right: 8px; top: 6px; }}
      .copy-btn, .user-btn {{
        font-size:13px;
        padding:6px 10px;
        border-radius:6px;
        border:1px solid #888;
        background:#fff;
        cursor:pointer;
      }}
      .user-btn {{ background:#f8f9fa; }}
      .bubble-content {{ white-space: pre-wrap; word-wrap: break-word; }}
    </style>
    </head>
    <body>
      <div id="chat" class="chat-wrap">
        {''.join(html_msgs)}
      </div>
      <script>
        const btns = document.querySelectorAll('.copy-btn, .user-btn');
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
              b.innerText = 'Failed';
              setTimeout(()=> b.innerText = 'Copy', 1200);
            }}
          }});
        }});
        const chat = document.getElementById('chat');
        chat.scrollTop = chat.scrollHeight;
      </script>
    </body>
    </html>
    """
    return html_doc

def make_pdf_bytes(title, conversation):
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    y -= 24
    c.setFont("Helvetica", 11)
    for turn in conversation:
        role = "You" if turn["role"]=="user" else "AI"
        text = f"{role}: {turn['content']}"
        for line in split_text_for_pdf(text, int(width - 2*margin), c):
            y -= 14
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 11)
            c.drawString(margin, y, line)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

def split_text_for_pdf(text, max_width_px, canvas_obj):
    words = text.split()
    lines = []
    line = ""
    for w in words:
        if len(line + " " + w) > 120:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    return lines

# ------------------ UI ------------------
st.title("AI-Driven Personalized Cancer Care Chatbot")
col1, col2 = st.columns([1, 0.6])
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
    manual_text = st.text_area("Or paste a short lab/test excerpt here (English, recommended):", height=140)

with col2:
    st.subheader("Controls & Mailgun helper")
    generate_clicked = st.button("Generate Summary & Dietary Advice")
    if MAILGUN_DOMAIN and MAILGUN_DOMAIN.startswith("sandbox"):
        st.info("Using Mailgun sandbox domain. Authorise recipients before sending email.")
        sandbox_link = f"https://app.mailgun.com/app/sending/domains/{MAILGUN_DOMAIN}"
        st.markdown(f"[Open Mailgun sandbox domain settings]({sandbox_link})")
    st.markdown(
        "- Responses are in English.\n"
        "- Dietary advice is dietitian-level and includes a 1-day sample menu.\n"
        "- Use the chat box below to refine; AI replies display on the left, your messages on the right."
    )

# ------------------ Generate handling ------------------
ocr_text = ""
if uploaded_files:
    ocr_text = ocr_extract(uploaded_files)
    if ocr_text:
        st.info(f"OCR extracted text (truncated to {MAX_OCR_CHARS} chars).")

if generate_clicked:
    prompt_text = manual_text.strip() or ocr_text.strip()
    if not prompt_text:
        st.warning("Please upload a report or paste text before generating.")
    else:
        messages = [{"role":"system","content":"You are an expert oncology dietitian and clinical-support assistant. Provide a concise clinical summary, three practical questions for the next doctor visit (non-dietary), and dietitian-level dietary advice including a clear 1-day sample menu separated by spacing. Use plain English."},
                    {"role":"user","content": prompt_text}]
        ai_text = call_openai_chat(messages)
        if ai_text:
            st.session_state.conversation.append({"role":"assistant","content": ai_text})

# ------------------ Chat input (bottom) ------------------
st.markdown("---")
st.subheader("Chat — refine advice or ask questions")
chat_input = st.text_input("Type follow-up or refinement here and press Send:", "")

send_clicked = st.button("Send")
if send_clicked and chat_input.strip():
    st.session_state.conversation.append({"role":"user","content": chat_input.strip()})
    messages = [{"role":"system","content":"You are an expert oncology dietitian and clinical-support assistant. Provide a concise clinical summary, three practical questions for the next doctor visit (non-dietary), and dietitian-level dietary advice including a clear 1-day sample menu separated by spacing. Use plain English."}]
    for turn in st.session_state.conversation:
        if turn["role"] == "user":
            messages.append({"role":"user","content": turn["content"]})
        else:
            messages.append({"role":"assistant","content": turn["content"]})
    ai_reply = call_openai_chat(messages)
    if ai_reply:
        st.session_state.conversation.append({"role":"assistant","content": ai_reply})

# ------------------ Render chat HTML ------------------
st.markdown("### Conversation (latest at bottom)")
chat_html = build_chat_html(st.session_state.conversation)
components.html(chat_html, height=560)

# ------------------ Download / Email ------------------
st.markdown("---")
st.subheader("Export / Email")

if st.button("Prepare Downloadable Report"):
    if not st.session_state.conversation:
        st.warning("No conversation available. Generate summary first.")
    else:
        title = f"AI Health Report - {datetime.datetime.now().strftime('%Y-%m-%d %H%M')}"
        if REPORTLAB_AVAILABLE:
            pdf_bytes = make_pdf_bytes(title, st.session_state.conversation)
            if pdf_bytes:
                st.download_button(label="Download PDF Report", data=pdf_bytes, file_name="ai_health_report.pdf", mime="application/pdf")
            else:
                text = "\n\n".join([f"{'You' if c['role']=='user' else 'AI'}: {c['content']}" for c in st.session_state.conversation])
                st.download_button(label="Download TXT Report", data=text, file_name="ai_health_report.txt", mime="text/plain")
        else:
            text = "\n\n".join([f"{'You' if c['role']=='user' else 'AI'}: {c['content']}" for c in st.session_state.conversation])
            st.download_button(label="Download TXT Report", data=text, file_name="ai_health_report.txt", mime="text/plain")
            st.info("Install reportlab in the environment to enable PDF export (optional).")

# Mailgun: Authorize recipient helper + send
st.markdown("---")
st.subheader("Mailgun: Authorize recipient (sandbox) / Send report")

cola, colb = st.columns([1,1])
with cola:
    email_to = st.text_input("Recipient email (for Mailgun):")
with colb:
    if MAILGUN_DOMAIN and MAILGUN_DOMAIN.startswith("sandbox"):
        if st.button("Authorize recipient (sandbox)"):
            if not email_to:
                st.warning("Enter recipient email to authorize.")
            else:
                res = authorize_mailgun_sandbox_recipient(email_to)
                if res["ok"]:
                    st.success("Authorized recipient. Check recipient inbox and click verification link.")
                else:
                    st.error("Authorization failed. Debug:")
                    st.write(res)
    if st.button("Send Report via Mailgun"):
        if not email_to:
            st.warning("Enter recipient email.")
        elif not st.session_state.conversation:
            st.warning("No conversation to send.")
        else:
            final_text = "\n\n".join([f"{'You' if c['role']=='user' else 'AI'}: {c['content']}" for c in st.session_state.conversation])
            result = send_mailgun_email(email_to, "AI Health Summary & Dietary Advice", final_text)
            if result["ok"]:
                st.success("Email sent successfully.")
            else:
                if result["status"] == 403:
                    st.error("Mailgun returned 403 Forbidden. If you use a sandbox domain, add the recipient as an authorised recipient in Mailgun (use the Authorize button above or add via Mailgun dashboard). If you use a custom domain, ensure DNS verification is complete.")
                else:
                    st.error("Failed to send email. Debug info below:")
                st.write(result)






















































