# app.py
import streamlit as st
import os
import re
from openai import OpenAI
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import easyocr
from twilio.rest import Client as TwilioClient

# ===== Initialize EasyOCR reader =====
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)  # GPU disabled for Streamlit Cloud

# ===== Streamlit page config =====
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("🧠 AI-Driven Personalized Cancer Care Chatbot")
st.write(
    "Upload medical reports or imaging files (JPG/PNG/PDF) or paste a short lab/test excerpt. "
    "Click Generate to get health summary, suggested doctor questions, nutrition advice and dietary deep-dive when needed."
)

# ===== OpenAI API key and client setup =====
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found in Streamlit Secrets. Paste below for this session:")
    api_key_input = st.text_input("OpenAI API Key:", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ===== Twilio Setup (Optional) =====
twilio_client = None
if "TWILIO_ACCOUNT_SID" in st.secrets and "TWILIO_AUTH_TOKEN" in st.secrets:
    try:
        twilio_client = TwilioClient(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
    except Exception as e:
        st.error(f"Twilio client init error: {e}")

# ===== Input UI =====
st.subheader("1) Input medical reports or lab summary")
uploaded_files = st.file_uploader(
    "Upload medical reports / imaging files (JPG/PNG/PDF). You can upload multiple files.",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)
text_input = st.text_area("Or paste a short lab/test excerpt here", height=160)

# ===== OCR Extraction =====
lab_texts = []
image_texts = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type.startswith("image"):
                img = Image.open(uploaded_file)
                st.image(img, caption=f"Preview: {uploaded_file.name}", use_column_width=True)
                ocr_result = "\n".join(reader.readtext(np.array(img), detail=0))
            elif uploaded_file.type == "application/pdf":
                pages = convert_from_bytes(uploaded_file.read())
                ocr_result = ""
                for page in pages:
                    ocr_result += "\n".join(reader.readtext(np.array(page), detail=0)) + "\n"
            else:
                st.warning(f"{uploaded_file.name} is not a supported file type.")
                continue

            fname_lower = uploaded_file.name.lower()
            if any(k in fname_lower for k in ["pet", "ct", "xray", "scan"]):
                image_texts.append(ocr_result)
            else:
                lab_texts.append(ocr_result)

        except Exception as e:
            st.error(f"OCR failed for {uploaded_file.name}: {e}")

# If OCR found text, show editable text area
if lab_texts:
    text_input = st.text_area("OCR extracted lab text (editable)", value="\n".join(lab_texts), height=200)

input_source = text_input.strip() if text_input and text_input.strip() else None

# ===== Health Index Calculation (unchanged) =====
def compute_health_index_smart(report_text):
    score = 50
    keywords_positive = ["normal", "stable", "remission", "improved"]
    keywords_negative = ["metastasis", "high", "low", "elevated", "decreased", "critical", "abnormal", "progression"]
    text_lower = report_text.lower()
    for kw in keywords_positive:
        if kw in text_lower:
            score += 5
    for kw in keywords_negative:
        if kw in text_lower:
            score -= 5
    return max(0, min(100, score))

def compute_health_index_with_imaging(report_texts, image_reports_texts=None):
    lab_score = compute_health_index_smart(report_texts) if report_texts else 50
    image_score = 100
    if image_reports_texts:
        deductions = []
        for text in image_reports_texts:
            text_lower = text.lower()
            for kw in ["metastasis", "lesion", "tumor growth", "progression"]:
                if kw in text_lower:
                    deductions.append(10)
            for kw in ["stable", "no abnormality", "remission", "no evidence of disease"]:
                if kw in text_lower:
                    deductions.append(-5)
        total_deduction = sum(deductions)
        image_score = max(0, min(100, image_score - total_deduction))
    combined_score = lab_score * 0.7 + image_score * 0.3
    return round(combined_score, 1)

# ===== New: Lab parsing helper =====
def parse_lab_values(text):
    """
    Try to extract numeric lab values commonly used:
      - Hemoglobin (g/dL)
      - WBC (10^9/L)
      - Neutrophils (absolute or %)
      - Platelets (10^9/L)
      - Glucose (mmol/L or mg/dL)
    Returns a dict of extracted floats or None.
    """
    if not text:
        return {}
    t = text.lower()
    results = {}

    # Helper to search patterns with optional units and commas
    def find_one(patterns):
        for p in patterns:
            m = re.search(p, t, flags=re.IGNORECASE)
            if m:
                # try group 1 or 2
                for g in (1,2,3):
                    try:
                        val = m.group(g)
                        if val:
                            # remove commas
                            val = val.replace(",", "")
                            # extract first numeric sequence
                            num = re.search(r"[-+]?\d*\.?\d+", val)
                            if num:
                                return float(num.group(0))
                    except Exception:
                        continue
        return None

    # Hemoglobin (g/dL)
    results['hb_g_dl'] = find_one([r"hemoglobin[:\s]*([\d\.]+)", r"hgb[:\s]*([\d\.]+)"])

    # WBC (10^9/L) or cells per mm3 — common formats
    results['wbc'] = find_one([
        r"wbc[:\s]*([\d\.]+)", 
        r"white blood cell[s]?:[:\s]*([\d\.]+)",
        r"wbc\s*count[:\s]*([\d\.]+)"
    ])

    # Neutrophils absolute (if absolute given) or percent (if percent given)
    results['neutrophil_abs'] = find_one([
        r"neutrophil[s]?\s*(?:absolute)?[:\s]*([\d\.]+)",
        r"neutrophil count[:\s]*([\d\.]+)"])
    # neutrophil% 
    neut_percent = find_one([r"neutrophil[s]?\s*%\s*[:\s]*([\d\.]+)", r"neutrophil[s]?\s*percent[:\s]*([\d\.]+)"])
    if neut_percent and results.get('wbc'):
        # convert percent -> absolute (approx)
        try:
            results['neutrophil_abs_calculated'] = (neut_percent/100.0) * results['wbc']
        except:
            results['neutrophil_abs_calculated'] = None

    # Platelets
    results['plt'] = find_one([r"platelet[s]?:[:\s]*([\d\.]+)", r"plt[:\s]*([\d\.]+)"])

    # Glucose (could be mmol/L or mg/dL)
    results['glucose'] = find_one([r"glucose[:\s]*([\d\.]+)", r"fasting glucose[:\s]*([\d\.]+)"])

    return results

# ===== New: Dietary Deep Dive generator (EN + CN) =====
def generate_dietary_deep_dive(lab_vals):
    """
    Given parsed lab values, produce a targeted dietary deep dive.
    Returns dict with 'cn' and 'en' keys containing text.
    """
    cn_lines = []
    en_lines = []

    cn_lines.append("【诊断提示与总览】")
    en_lines.append("【Clinical note & overview】")

    # Example: neutropenia / low WBC
    wbc = lab_vals.get('wbc')
    neut = lab_vals.get('neutrophil_abs') or lab_vals.get('neutrophil_abs_calculated')
    hb = lab_vals.get('hb_g_dl')
    plt = lab_vals.get('plt')
    glu = lab_vals.get('glucose')

    neutropenia_flag = False
    if neut is not None:
        # neut absolute often in 10^9/L; thresholds: <1.5 mild, <1.0 moderate, <0.5 severe
        if neut < 1.5:
            neutropenia_flag = True
            if neut < 0.5:
                severity = "severe"
            elif neut < 1.0:
                severity = "moderate"
            else:
                severity = "mild"
        else:
            severity = None
    elif wbc is not None:
        # fallback: if total WBC low
        if wbc < 3.0:
            neutropenia_flag = True
            severity = "possible (WBC low)"
        else:
            severity = None
    else:
        severity = None

    if neutropenia_flag:
        cn_lines.append(f"患者存在中性粒细胞/白细胞减少（严重度标注: {severity}）。这会增加感染风险，需采取饮食与食品安全的额外防护。")
        en_lines.append(f"Patient shows neutropenia / low WBC (severity: {severity}). Infection risk is elevated — dietary precautions and food-safety measures are important.")
        # Practical food guidance (specific to neutropenia)
        cn_lines.append("实务建议（中性粒细胞低）:")
        en_lines.append("Practical guidance (neutropenia):")

        cn_lines.extend([
            "- 优先高质量熟蛋白（煮熟的鸡蛋、煮熟鱼、去皮鸡胸肉、豆腐、希腊优格）。",
            "- 增加能产生短链脂肪酸（SCFA）的熟谷物与可溶性纤维，例如熟燕麦、熟糙米、熟香蕉与燕麦麸（每日适量1–2份）。",
            "- 选择巴氏灭菌或高温处理过的乳制品，避免生食奶与生蛋食品。", 
            "- 避免生海鲜、生菜沙拉、生芽菜、未彻底煮熟的肉类与街边未加热熟食。", 
            "- 适量提供含锌与硒的食物（如一小把南瓜籽、少量巴西果），但避免过量补充（补充剂须先询问医生）。",
            "- 多喝温开水，确保口腔卫生，若出现发烧立刻就医。"
        ])
        en_lines.extend([
            "- Prioritise well-cooked high-quality proteins: hard-boiled/fully cooked eggs, cooked fish, skinless chicken breast, tofu, pasteurised yogurt.",
            "- Increase cooked whole grains and soluble fiber that support SCFA (e.g., cooked oats, cooked brown rice, cooked banana, oat bran) — aim for modest servings (1–2 servings/day).",
            "- Use pasteurised or heat-treated dairy; avoid raw milk/soft cheeses and raw-egg dishes.",
            "- Avoid raw seafood, raw salads, raw sprouts, undercooked meats and street foods that are not reheated.",
            "- Include zinc/selenium containing foods (small handful pumpkin seeds, small amount Brazil nuts) but avoid high-dose supplements without doctor approval.",
            "- Maintain hydration and oral hygiene. If fever occurs, seek medical attention immediately."
        ])

        # Example specific nutrition rationale
        cn_lines.append("营养学理由:")
        en_lines.append("Nutritional rationale:")
        cn_lines.append("- 熟谷物与可溶性纤维通过肠道细菌产生短链脂肪酸（SCFA），可支持肠道屏障与免疫功能。")
        en_lines.append("- Cooked whole grains and soluble fiber support SCFA production by the gut microbiome, which helps gut barrier and immune resilience.")

    # Additional rules: anemia, thrombocytopenia, glucose
    if hb is not None and hb < 12:
        cn_lines.append("贫血相关建议:")
        en_lines.append("Anemia-related suggestions:")
        cn_lines.append("- 增加含铁与高质量蛋白的食物（瘦红肉、鸡肝适量、豆类、菠菜与南瓜籽），搭配维生素C（如蒸红椒或柑橘）以帮助吸收。")
        en_lines.append("- Increase iron and high-quality protein sources (lean red meat, small amounts liver if approved, legumes, spinach, pumpkin seeds) paired with vitamin C to aid absorption.")
        cn_lines.append("- 若为化疗相关贫血，请在医生建议下考虑铁剂或促红素治疗。")
        en_lines.append("- For chemo-related anemia, discuss iron therapy or erythropoiesis support with physician.")

    if plt is not None and plt < 100:
        cn_lines.append("血小板较低（出血风险）注意:")
        en_lines.append("Low platelets (bleeding risk) notes:")
        cn_lines.append("- 避免硬脆、容易划伤牙龈的食物（如坚果粗碎直接咀嚼，需改为细磨或切小块）。")
        en_lines.append("- Avoid hard, sharp foods that can injure oral mucosa; modify textures.")

    if glu is not None:
        cn_lines.append("血糖注意:")
        en_lines.append("Glucose notes:")
        if glu > 7.0:
            cn_lines.append("- 血糖偏高，推荐减少精制糖与含糖饮料，增加低升糖指数字食物（全谷、豆类、蔬菜）。")
            en_lines.append("- Hyperglycaemia present: reduce refined sugars/drinks; prefer low-GI foods such as whole grains, legumes, vegetables.")
        else:
            cn_lines.append("- 血糖在可接受范围，保持均衡碳水与蛋白质摄入以维持能量。")
            en_lines.append("- Glucose within acceptable range; maintain balanced carbs and protein.")

    # Practical one-day sample (short)
    cn_lines.append("示例一日餐单（供参考）:")
    en_lines.append("Sample 1-day menu (for reference):")
    cn_lines.append("- 早餐：熟燕麦粥 + 熟香蕉切片 + 一小把南瓜籽 + 巴氏酸奶。")
    cn_lines.append("- 午餐：蒸熟鸡胸肉 + 熟糙米 + 蒸红萝卜 + 小份拌熟菠菜（加柠檬）。")
    cn_lines.append("- 晚餐：清蒸鱼 + 熟藜麦/糙米 + 蒸绿叶菜。")
    cn_lines.append("- 小食：煮蛋一枚（全熟）、少量水果（熟苹果泥）。")
    en_lines.append("- Breakfast: cooked oats + sliced cooked banana + small handful pumpkin seeds + pasteurised yogurt.")
    en_lines.append("- Lunch: steamed chicken breast + cooked brown rice + steamed carrot + small serving cooked spinach with lemon.")
    en_lines.append("- Dinner: steamed fish + cooked quinoa/brown rice + steamed greens.")
    en_lines.append("- Snacks: hard-boiled egg (fully cooked), small portion cooked fruit compote.")

    cn_lines.append("重要警告:")
    en_lines.append("Important cautions:")
    cn_lines.append("- 若患者处于严重免疫抑制或正接受化疗，请勿给任何未煮熟或生食，所有食材以彻底加热为主。")
    cn_lines.append("- 在开始任何补剂（如高剂量锌、维生素D或抗氧化剂）前务必与主治医师确认，避免影响化疗或药物代谢。")
    en_lines.append("- If severely immunosuppressed or on chemotherapy, avoid all raw/undercooked foods; heat everything thoroughly.")
    en_lines.append("- Discuss supplements (high-dose zinc, vitamin D, antioxidants) with treating physician before starting.")

    # Combine
    cn_text = "\n".join(cn_lines)
    en_text = "\n".join(en_lines)
    return {"cn": cn_text, "en": en_text, "flag": ("neutropenia" if neutropenia_flag else None)}

# ===== Helper: robust section extractor (unchanged) =====
def extract_section(text, header):
    pattern = rf"{header}\s*[:\-]?\s*(.*?)(?=\n(?:Summary|Questions|Nutrition)\s*[:\-]|\Z)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    variants = {
        "Questions": [r"questions to ask", r"doctor questions", r"questions:", r"questions to ask the doctor", r"questions for doctor"],
        "Summary": [r"summary", r"health summary", r"clinical summary"],
        "Nutrition": [r"nutrition", r"recommendations", r"diet", r"nutrition recommendations"]
    }
    for v in variants.get(header, []):
        m2 = re.search(rf"{v}\s*[:\-]?\s*(.*?)(?=\n(?:summary|questions|nutrition)\s*[:\-]|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        if m2:
            return m2.group(1).strip()
    return "No findings."

# ===== Function: Twilio pre-check + send helper (unchanged) =====
def twilio_send_with_precheck(client, from_, to, test_body, real_body):
    try:
        test_msg = client.messages.create(body=test_body, from_=from_, to=to)
        if getattr(test_msg, "error_code", None):
            return False, test_msg
        real_msg = client.messages.create(body=real_body, from_=from_, to=to)
        return True, real_msg
    except Exception as e:
        return False, e

# ===== Button: Generate AI output =====
if st.button("Generate Summary & Recommendations"):
    if not input_source and not lab_texts:
        st.error("Please paste a lab/test excerpt or upload OCR-compatible files first.")
    elif not client:
        st.error("OpenAI client not configured. Please set OPENAI_API_KEY in Streamlit Secrets or paste above.")
    else:
        with st.spinner("Generating AI output..."):
            all_lab_text = input_source if input_source else "\n".join(lab_texts)
            health_index = compute_health_index_with_imaging(all_lab_text, image_texts)

            # save to session_state early
            st.session_state['health_index'] = health_index

            st.subheader("📊 Health Index")
            st.write(f"Combined Health Index (0-100): {health_index}")

            # GPT Prompt (more strict)
            prompt = f"""
You are a clinical-support assistant. Given the patient's report text below, produce exactly three labelled sections: Summary, Questions, Nutrition.
- Write "Summary:" then 3-4 short sentences in plain language.
- Write "Questions:" then a numbered or bulleted list of three practical questions the patient/family should ask the doctor next visit.
- Write "Nutrition:" then three simple, food-based nutrition recommendations based on Malaysia Cancer Nutrition Guidelines.
If any section has no data to provide, write the section header and then "No findings.".

Patient report:
\"\"\"{all_lab_text}\"\"\"

Output format (must include these headers exactly): 
Summary:
- ...
Questions:
- ...
Nutrition:
- ...
"""
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=700,
                    temperature=0.2
                )
                try:
                    ai_text = resp.choices[0].message.content
                except Exception:
                    ai_text = resp["choices"][0]["message"]["content"] if "choices" in resp else str(resp)

                # parse sections
                summary = extract_section(ai_text, "Summary")
                questions = extract_section(ai_text, "Questions")
                nutrition = extract_section(ai_text, "Nutrition")

                # normalize
                if not summary or summary.strip() == "":
                    summary = "No findings."
                if not questions or questions.strip() == "":
                    questions = "No findings."
                if not nutrition or nutrition.strip() == "":
                    nutrition = "No findings."

                # save to session_state
                st.session_state['summary'] = summary
                st.session_state['questions'] = questions
                st.session_state['nutrition'] = nutrition
                st.session_state['ai_raw'] = ai_text

                # Display
                st.subheader("🧾 Health Summary")
                st.write(summary)
                st.subheader("❓ Suggested Questions for the Doctor")
                st.write(questions)
                st.subheader("🥗 Nutrition Recommendations")
                st.write(nutrition)

                # ===== New: Parse numeric labs and show Dietary Deep Dive if needed =====
                labs = parse_lab_values(all_lab_text)
                st.write("🔬 Parsed lab values (automated):", labs)
                deep = generate_dietary_deep_dive(labs)
                if deep.get("flag"):
                    st.subheader("🧾 Dietary Deep Dive (targeted)")
                    st.write("（系统检测到高风险指标，已展开更详细的饮食与食品安全建议）")
                    # show both CN and EN with expanders
                    with st.expander("中文 — 深度饮食建议 (可直接复制给家人)"):
                        st.text_area("Copyable Chinese Deep Dive", value=deep['cn'], height=300)
                    with st.expander("English — Dietary Deep Dive (copyable)"):
                        st.text_area("Copyable English Deep Dive", value=deep['en'], height=300)
                    # save into session_state for sending/emailing
                    st.session_state['deep_cn'] = deep['cn']
                    st.session_state['deep_en'] = deep['en']
                else:
                    st.info("No immediate dietary deep-dive flags detected (e.g., neutropenia). You can still request more detailed dietary advice manually.")

                with st.expander("Full AI output (raw)"):
                    st.code(ai_text)

            except Exception as e:
                st.error(f"OpenAI API call failed: {e}")

# ===== Button: Send WhatsApp with pre-check (sandbox-friendly) =====
if twilio_client and st.button("Send Health Update to Family via WhatsApp (with pre-check)"):
    if 'health_index' not in st.session_state:
        st.error("Generate AI output first before sending WhatsApp message.")
    else:
        from_num = st.secrets.get("TWILIO_WHATSAPP_FROM")
        to_num = st.secrets.get("TWILIO_WHATSAPP_TO")
        if not from_num or not to_num:
            st.error("TWILIO_WHATSAPP_FROM or TWILIO_WHATSAPP_TO not set in secrets.")
        else:
            test_body = "Twilio Sandbox check — please ensure you have sent the join code to +14155238886."
            real_body = (
                f"Health Index: {st.session_state.get('health_index','N/A')}\n\n"
                f"Summary:\n{st.session_state.get('summary','No findings.')}\n\n"
                f"Questions:\n{st.session_state.get('questions','No findings.')}\n\n"
                f"Nutrition:\n{st.session_state.get('nutrition','No findings.')}\n\n"
                f"Dietary Deep Dive (EN):\n{st.session_state.get('deep_en','No deep dive.')}"
            )
            with st.spinner("Running Sandbox pre-check and sending..."):
                ok, detail = twilio_send_with_precheck(twilio_client, from_num, to_num, test_body, real_body)
                if ok:
                    st.success("Test OK — real message sent.")
                    st.write("Message SID:", getattr(detail, "sid", None))
                    st.write("Status:", getattr(detail, "status", None))
                    st.write("Date created:", getattr(detail, "date_created", None))
                else:
                    st.error("Failed to send. See debug info below.")
                    if not isinstance(detail, Exception):
                        st.write("Twilio response object:", detail)
                        st.write("error_code:", getattr(detail, "error_code", None))
                        st.write("error_message:", getattr(detail, "error_message", None))
                        st.write("status:", getattr(detail, "status", None))
                    else:
                        st.write("Exception type:", type(detail))
                        try:
                            st.write("Exception repr:", repr(detail))
                            st.write("Exception args:", detail.args)
                        except:
                            st.write("Could not extract exception details.")
                    st.info("If error_code indicates 'number not joined' (e.g. 63016), please:")
                    st.write("1. On your phone, send the join code (shown in Twilio Console → Messaging → Try WhatsApp Sandbox) to +14155238886.")
                    st.write("2. Confirm you used the same WhatsApp account/phone number that is set as TWILIO_WHATSAPP_TO in secrets.")
                    st.write("3. After successful join confirmation message on your phone, re-run this Send action.")























