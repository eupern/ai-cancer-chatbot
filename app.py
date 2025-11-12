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
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# ===== Streamlit page config =====
st.set_page_config(page_title="AI-Driven Personalized Cancer Care Chatbot", layout="centered")
st.title("🧠 AI-Driven Personalized Cancer Care Chatbot")
st.write(
    "Upload medical reports (JPG/PNG/PDF) or paste a short lab/test excerpt. Click Generate to get health summary, questions for doctor, nutrition advice and (when flagged) a Dietary Deep Dive."
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

# ===== Twilio (optional) =====
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
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption=f"Preview: {uploaded_file.name}", use_column_width=True)
                ocr_result = "\n".join(reader.readtext(np.array(img), detail=0))
            elif uploaded_file.type == "application/pdf":
                pages = convert_from_bytes(uploaded_file.read())
                ocr_result = ""
                for page in pages:
                    page_arr = np.array(page.convert("RGB"))
                    ocr_result += "\n".join(reader.readtext(page_arr, detail=0)) + "\n"
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

# ===== Health Index Calculation =====
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

# ===== Lab parsing helper =====
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
                            val = val.replace(",", "")
                            num = re.search(r"[-+]?\d*\.?\d+", val)
                            if num:
                                return float(num.group(0))
                    except Exception:
                        continue
        return None

    results['hb_g_dl'] = find_one([r"hemoglobin[:\s]*([\d\.]+)", r"hgb[:\s]*([\d\.]+)"])
    results['wbc'] = find_one([r"wbc[:\s]*([\d\.]+)", r"white blood cell[s]?:[:\s]*([\d\.]+)", r"wbc count[:\s]*([\d\.]+)"])
    results['neutrophil_abs'] = find_one([r"neutrophil[s]?\s*(?:absolute)?[:\s]*([\d\.]+)", r"neutrophil count[:\s]*([\d\.]+)"])
    neut_percent = find_one([r"neutrophil[s]?\s*%\s*[:\s]*([\d\.]+)", r"neutrophil[s]?\s*percent[:\s]*([\d\.]+)"])
    if neut_percent and results.get('wbc'):
        try:
            results['neutrophil_abs_calculated'] = (neut_percent / 100.0) * results['wbc']
        except:
            results['neutrophil_abs_calculated'] = None
    results['plt'] = find_one([r"platelet[s]?:[:\s]*([\d\.]+)", r"plt[:\s]*([\d\.]+)"])
    results['glucose'] = find_one([r"glucose[:\s]*([\d\.]+)", r"fasting glucose[:\s]*([\d\.]+)"])
    return results

# ===== Dietary Deep Dive content generator =====
def generate_dietary_deep_dive(lab_vals):
    cn_lines = []
    en_lines = []

    cn_lines.append("【诊断提示与总览】")
    en_lines.append("【Clinical note & overview】")

    wbc = lab_vals.get('wbc')
    neut = lab_vals.get('neutrophil_abs') or lab_vals.get('neutrophil_abs_calculated')
    hb = lab_vals.get('hb_g_dl')
    plt = lab_vals.get('plt')
    glu = lab_vals.get('glucose')

    neutropenia_flag = False
    severity = None
    if neut is not None:
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
        if wbc < 3.0:
            neutropenia_flag = True
            severity = "possible (WBC low)"
        else:
            severity = None

    if neutropenia_flag:
        cn_lines.append(f"患者存在中性粒细胞/白细胞减少（严重度: {severity}）。感染风险增加，请采取食品安全措施与饮食调整。")
        en_lines.append(f"Patient shows neutropenia / low WBC (severity: {severity}). Infection risk is elevated — follow food-safety measures and specific dietary choices.")
        cn_lines.append("实务建议（中性粒细胞低）:")
        en_lines.append("Practical guidance (neutropenia):")

        cn_lines.extend([
            "- 优先高质量熟蛋白（煮熟的鸡蛋、熟鱼、去皮鸡胸肉、豆腐、巴氏酸奶）。",
            "- 增加熟谷物和可溶性纤维以支持短链脂肪酸（SCFA）产生：熟燕麦、熟糙米、熟香蕉、燕麦麸（每日1–2份）。",
            "- 选择巴氏灭菌或经热处理的乳制品，避免生奶与生蛋制品。",
            "- 避免生海鲜、生菜沙拉、生芽菜、未彻底煮熟的肉类与不明来源街边熟食。",
            "- 适量摄入含锌/硒食物（如南瓜籽、小量巴西果），补剂须先询问医生。",
            "- 保持口腔卫生与充足饮水，若出现发热请立即就医。"
        ])
        en_lines.extend([
            "- Prioritise well-cooked, high-quality proteins: hard-boiled/fully cooked eggs, cooked fish, skinless chicken breast, tofu, pasteurised yogurt.",
            "- Increase cooked whole grains and soluble fiber that support SCFA production: cooked oats, cooked brown rice, cooked banana, oat bran (modest servings 1–2/day).",
            "- Use pasteurised or heat-treated dairy; avoid raw milk and raw-egg dishes.",
            "- Avoid raw seafood, raw salads, raw sprouts, undercooked meats and uncertain street foods.",
            "- Include zinc/selenium foods (pumpkin seeds, small amount Brazil nuts); consult physician before supplements.",
            "- Maintain hydration and oral hygiene. Seek urgent care if fever occurs."
        ])

        cn_lines.append("营养学理由:")
        en_lines.append("Nutritional rationale:")
        cn_lines.append("- 熟谷物与可溶性纤维通过肠道菌群产生短链脂肪酸（SCFA），有助于肠道屏障与免疫功能。")
        en_lines.append("- Cooked whole grains and soluble fiber feed the gut microbiome to produce SCFA, supporting gut barrier and immune resilience.")

    if hb is not None and hb < 12:
        cn_lines.append("贫血相关建议:")
        en_lines.append("Anemia-related suggestions:")
        cn_lines.append("- 增加含铁与高质量蛋白的食物（瘦红肉、豆类、菠菜、南瓜籽），并搭配维生素C帮助吸收。")
        en_lines.append("- Increase iron and quality protein sources (lean red meat, legumes, spinach, pumpkin seeds) paired with vitamin C.")

    if plt is not None and plt < 100:
        cn_lines.append("血小板较低提示（出血风险）:")
        en_lines.append("Low platelets (bleeding risk) notes:")
        cn_lines.append("- 避免硬脆或易划伤口腔的食物；将坚果等切碎或磨粉食用以降低创伤风险。")
        en_lines.append("- Avoid hard, sharp foods; modify texture (chop or grind nuts).")

    if glu is not None:
        cn_lines.append("血糖注意:")
        en_lines.append("Glucose notes:")
        if glu > 7.0:
            cn_lines.append("- 血糖偏高，减少精制糖與含糖饮料，优先全谷与蔬菜。")
            en_lines.append("- Hyperglycaemia present: reduce refined sugars/drinks; prefer whole grains and vegetables.")
        else:
            cn_lines.append("- 血糖在可接受范围，保持均衡碳水與蛋白质摄入。")
            en_lines.append("- Glucose within acceptable range; maintain balanced carbs and protein.")

    cn_lines.append("示例一日餐单（参考）:")
    en_lines.append("Sample 1-day menu (reference):")
    cn_lines.append("- 早餐：熟燕麦粥 + 熟香蕉切片 + 少量南瓜籽 + 巴氏酸奶。")
    cn_lines.append("- 午餐：蒸鸡胸肉 + 熟糙米 + 蒸胡萝卜 + 熟菠菜。")
    cn_lines.append("- 晚餐：清蒸鱼 + 熟藜麦/糙米 + 蒸绿叶菜。")
    cn_lines.append("- 小食：全熟水煮蛋、少量熟水果泥。")
    en_lines.append("- Breakfast: cooked oats + cooked banana + pumpkin seeds + pasteurised yogurt.")
    en_lines.append("- Lunch: steamed chicken breast + cooked brown rice + steamed carrot + cooked spinach.")
    en_lines.append("- Dinner: steamed fish + cooked quinoa/brown rice + steamed greens.")
    en_lines.append("- Snacks: hard-boiled egg, small portion cooked fruit compote.")

    cn_lines.append("重要警告:")
    en_lines.append("Important cautions:")
    cn_lines.append("- 若患者处於严重免疫抑制或接受化疗，请避免所有生食与未煮熟食品，均以彻底加热为主。")
    cn_lines.append("- 在开始任何补剂（如高剂量锌、维生素D或抗氧化剂）前务必与主治医师确认。")
    en_lines.append("- If severely immunosuppressed or on chemotherapy, avoid all raw/undercooked foods; heat thoroughly.")
    en_lines.append("- Discuss supplements (high-dose zinc, vitamin D, antioxidants) with the treating physician before starting.")

    cn_text = "\n".join(cn_lines)
    en_text = "\n".join(en_lines)
    return {"cn": cn_text, "en": en_text, "flag": ("neutropenia" if neutropenia_flag else None)}

# ===== Robust section extractor =====
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

# ===== Twilio pre-check helper =====
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
            st.session_state['health_index'] = health_index

            st.subheader("📊 Health Index")
            st.write(f"Combined Health Index (0-100): {health_index}")

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

                summary = extract_section(ai_text, "Summary")
                questions = extract_section(ai_text, "Questions")
                nutrition = extract_section(ai_text, "Nutrition")

                if not summary or summary.strip() == "":
                    summary = "No findings."
                if not questions or questions.strip() == "":
                    questions = "No findings."
                if not nutrition or nutrition.strip() == "":
                    nutrition = "No findings."

                st.session_state['summary'] = summary
                st.session_state['questions'] = questions
                st.session_state['nutrition'] = nutrition
                st.session_state['ai_raw'] = ai_text

                st.subheader("🧾 Health Summary")
                st.write(summary)
                st.subheader("❓ Suggested Questions for the Doctor")
                st.write(questions)
                st.subheader("🥗 Nutrition Recommendations")
                st.write(nutrition)

                # Parse labs and possibly show Dietary Deep Dive
                labs = parse_lab_values(all_lab_text)
                st.write("🔬 Parsed lab values (automated):", labs)
                deep = generate_dietary_deep_dive(labs)
                if deep.get("flag"):
                    st.subheader("🍚 Dietary Deep Dive (targeted)")
                    st.write("System detected risk flags; detailed dietary & food-safety guidance is below (copyable).")
                    with st.expander("Chinese — Deep Dive (copyable)"):
                        st.text_area("Chinese Deep Dive", value=deep['cn'], height=300)
                    with st.expander("English — Deep Dive (copyable)"):
                        st.text_area("English Deep Dive", value=deep['en'], height=300)
                    st.session_state['deep_cn'] = deep['cn']
                    st.session_state['deep_en'] = deep['en']
                else:
                    st.info("No immediate dietary deep-dive flags detected. You can request more detailed dietary advice manually.")

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
























