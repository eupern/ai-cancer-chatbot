# app.py - improved: downscale images, OCR truncation, last-N history, timing
import streamlit as st
import numpy as np
from PIL import Image
import io
import time
import easyocr
import openai
import requests

# ------------------ CONFIG / SECRETS ------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]
MAILGUN_API_KEY = st.secrets.get("MAILGUN_API_KEY")
MAILGUN_DOMAIN = st.secrets.get("MAILGUN_DOMAIN")
EMAIL_SENDER = f"postmaster@{MAILGUN_DOMAIN}" if MAILGUN_DOMAIN else None

# Limits / tuning
MAX_OCR_CHARS = 3000       # keep OCR output reasonably sized
DOWNSCALE_MAX_SIDE = 1280  # max side (px) when downscaling images for OCR
LAST_N_TURNS = 6           # when calling the model, send only the last N turns of conversation

# ------------------ INIT ------------------
st.set_page_config(page_title="AI Cancer Chatbot", layout="wide")
st.title("AI-Driven Personalized Cancer Care Chatbot")

# Pre-warm OCR reader once
reader = None
with st.spinner("Initialising OCR model (first run may take a few seconds)..."):
    try:
        reader = easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        st.warning("EasyOCR initialisation failed; OCR disabled. You can still paste text manually.")
        st.write(e)

# Ensure conversation list exists
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# ------------------ HELPERS ------------------
def downscale_image_file(uploaded_file, max_side=DOWNSCALE_MAX_SIDE):
    """Return BytesIO of downscaled JPEG (quality 80)."""
    try:
        data = uploaded_file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        return buf
    except Exception as e:
        # fallback: return original stream
        uploaded_file.seek(0)
        return io.BytesIO(uploaded_file.read())

def ocr_extract_from_files(files):
    """Run OCR on uploaded image files, downscaling before OCR, return combined truncated text."""
    if not reader or not files:
        return ""
    texts = []
    for f in files:
        try:
            # downscale for faster OCR
            ds_buf = downscale_image_file(f)
            img = Image.open(ds_buf).convert("RGB")
            arr = np.array(img)
            txts = reader.readtext(arr, detail=0)
            if txts:
                texts.append(" ".join(txts))
        except Exception as e:
            st.error(f"OCR failed for {getattr(f,'name','file')}: {e}")
    combined = " ".join(texts)
    if len(combined) > MAX_OCR_CHARS:
        combined = combined[:MAX_OCR_CHARS]
    return combined

def build_messages_for_model(system_prompt, conversation_history, user_new=None):
    """
    Build messages list for OpenAI: system + last N turns + optional new user message.
    conversation_history stores dicts with role 'user'|'assistant' and content.
    """
    messages = [{"role": "system", "content": system_prompt}]
    # take last N turns
    history = conversation_history[-LAST_N_TURNS:] if conversation_history else []
    for turn in history:
        # ensure roles match API ('user' or 'assistant')
        role = turn.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        messages.append({"role": role, "content": turn.get("content", "")})
    if user_new:
        messages.append({"role": "user", "content": user_new})
    return messages

def call_openai_chat(messages):
    """Call OpenAI chat completions (gpt-5-mini). Do not pass unsupported parameters."""
    try:
        resp = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=messages
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"AI generation failed: {e}")
        return None

def send_mailgun_email(to_email, subject, text):
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN):
        st.warning("Mailgun not configured in secrets.")
        return {"ok": False, "status": None, "detail": "Mailgun not configured."}
    try:
        resp = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={"from": EMAIL_SENDER, "to": [to_email], "subject": subject, "text": text},
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
col1, col2 = st.columns([1, 0.6])
with col1:
    uploaded_files = st.file_uploader("Upload medical reports (images/PDF). Multiple allowed.", type=["jpg","jpeg","png","pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.subheader("Uploaded preview")
        for f in uploaded_files:
            try:
                # Show original preview (do not downscale for display)
                img = Image.open(f)
                st.image(img, caption=f.name, use_column_width=True)
            except Exception:
                st.write(f.name)
    manual_text = st.text_area("Or paste a short lab/test excerpt here (English):", height=160)

with col2:
    st.subheader("Controls")
    st.markdown("- Dietary advice is dietitian-level and includes a 1-day sample menu.\n- Chat below refines the advice.")
    generate_btn = st.button("Generate Summary & Dietary Advice")
    st.markdown("Mailgun: ensure sandbox recipients are authorised if using a sandbox domain.")

# ------------------ Generate handling with timing ------------------
system_prompt = ("You are an expert oncology dietitian and clinical-support assistant. "
                 "Provide a concise clinical summary (3-4 lines), three practical questions to ask the doctor, "
                 "and a structured dietitian-level dietary advice with a clearly separated 1-day sample menu. Use plain English.")

if generate_btn:
    # get OCR text (downscaled + truncated)
    t0 = time.time()
    ocr_text = ocr_extract_from_files(uploaded_files) if uploaded_files else ""
    t1 = time.time()
    source_text = manual_text.strip() or ocr_text.strip()
    if not source_text:
        st.warning("Please upload a report or paste text before generating.")
    else:
        messages = build_messages_for_model(system_prompt, st.session_state.conversation, user_new=source_text)
        t_api_start = time.time()
        ai_text = call_openai_chat(messages)
        t_api_end = time.time()
        if ai_text:
            # append as assistant message; keep conversation as alternating user/assistant entries
            # store the user prompt (source_text) then assistant reply for traceability
            st.session_state.conversation.append({"role": "user", "content": "(source) " + (manual_text.strip() or "[uploaded files]")})
            st.session_state.conversation.append({"role": "assistant", "content": ai_text})
            st.success("AI response generated and added to chat.")
            st.info(f"OCR took {t1-t0:.2f}s; OpenAI call took {t_api_end-t_api_start:.2f}s (total {t_api_end-t0:.2f}s).")

# ------------------ Chat input for follow-up (bottom area) ------------------
st.markdown("---")
st.subheader("Chat — refine advice or ask follow-up questions")
chat_input = st.text_input("Type a follow-up or refinement and press Send:", "")

if st.button("Send"):
    if chat_input.strip():
        # record user turn
        st.session_state.conversation.append({"role": "user", "content": chat_input.strip()})
        # build messages with last N turns + new user prompt
        messages = build_messages_for_model(system_prompt, st.session_state.conversation, user_new=chat_input.strip())
        t_api_start = time.time()
        ai_reply = call_openai_chat(messages)
        t_api_end = time.time()
        if ai_reply:
            st.session_state.conversation.append({"role": "assistant", "content": ai_reply})
            st.info(f"OpenAI call took {t_api_end-t_api_start:.2f}s.")

# ------------------ Display conversation as simple chat (left=AI, right=You) ------------------
st.subheader("Conversation (latest at bottom)")
for msg in st.session_state.conversation:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    if role == "assistant":
        st.markdown(f"**AI:** {content}")
    else:
        st.markdown(f"**You:** {content}")

# ------------------ Email send ------------------
st.markdown("---")
st.subheader("Send final report via Mailgun (optional)")
email_to = st.text_input("Recipient email (for Mailgun):")
if st.button("Send Email"):
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
                st.error("Mailgun 403 Forbidden. If using sandbox domain, add the recipient as authorised in Mailgun dashboard.")
            else:
                st.error("Failed to send email. Debug info:")
            st.write(result)
























































