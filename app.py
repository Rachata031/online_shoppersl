"""
Online Shoppers Purchase Intention — Streamlit App
โมเดล: Random Forest (random_forest_online_shoppers.joblib หรือ .zip)
"""

import inspect
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# ค่าคงที่
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "random_forest_online_shoppers.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

APP_TITLE = "ระบบทำนายโอกาสการซื้อสินค้าของผู้เข้าชมเว็บไซต์"
APP_SUBTITLE = "Online Shoppers Purchase Intention · Random Forest"
DEVELOPERS = "รชต ทัดทอง : วารุณี มากทิพย์ : อัญชิษา ปิ่นสุข"

# โมเดลถูกเทรนด้วยข้อมูลที่ปรับสเกลแบบ MinMax (0-1) จึงต้องปรับสเกลอินพุตก่อนทำนาย
# ค่าสูงสุดของแต่ละฟีเจอร์ (ค่าต่ำสุดคือ 0 ทั้งหมด)
SCALE_MAX = {
    "Administrative": 27,
    "Administrative_Duration": 3398.75,
    "Informational": 24,
    "Informational_Duration": 2549.375,
    "ProductRelated": 705,
    "ProductRelated_Duration": 63973.52222,
    "BounceRates": 0.2,
    "ExitRates": 0.2,
    "PageValues": 361.7637419,
    "TotalPages": 746,
    "TotalDuration": 3398.75 + 2549.375 + 63973.52222,
}

# เดือนในชุดข้อมูล (สิงหาคมเป็นค่าอ้างอิง คือทุกคอลัมน์ Month_* = 0)
MONTHS = {
    "กุมภาพันธ์": "Feb",
    "มีนาคม": "Mar",
    "พฤษภาคม": "May",
    "มิถุนายน": "June",
    "กรกฎาคม": "Jul",
    "สิงหาคม": "Aug",
    "กันยายน": "Sep",
    "ตุลาคม": "Oct",
    "พฤศจิกายน": "Nov",
    "ธันวาคม": "Dec",
}

# ผู้เข้าชมใหม่เป็นค่าอ้างอิง (VisitorType_* = 0 ทั้งคู่)
VISITOR_TYPES = {
    "ผู้เข้าชมใหม่ (ครั้งแรก)": "New_Visitor",
    "ผู้เข้าชมเดิม (เคยเข้ามาแล้ว)": "Returning_Visitor",
    "อื่น ๆ": "Other",
}
WEEKDAY, WEEKEND = "วันธรรมดา", "วันหยุดสุดสัปดาห์"

# ค่าเริ่มต้นของฟอร์ม (หน่วยตามที่แสดงบนหน้าจอ: เวลาเป็นนาที, อัตราเป็น %)
DEFAULTS = {
    "admin": 0, "admin_min": 0.0,
    "info": 0, "info_min": 0.0,
    "prod": 10, "prod_min": 5.0,
    "bounce_pct": 2.0, "exit_pct": 4.0,
    "page_value": 0.0, "special": 0.0,
    "month": "พฤศจิกายน",
    "visitor": "ผู้เข้าชมเดิม (เคยเข้ามาแล้ว)",
    "weekend": WEEKDAY,
    "os": 2, "browser": 2, "region": 1, "traffic": 2,
}

# ตัวอย่างสำเร็จรูป เพื่อให้ลองใช้งานได้ทันที
PRESETS = {
    "interested": {
        "admin": 2, "admin_min": 2.0, "info": 0, "info_min": 0.0,
        "prod": 30, "prod_min": 15.0,
        "bounce_pct": 0.5, "exit_pct": 1.5, "page_value": 30.0, "special": 0.0,
        "month": "พฤศจิกายน", "visitor": "ผู้เข้าชมเดิม (เคยเข้ามาแล้ว)", "weekend": WEEKDAY,
    },
    "browsing": {
        "admin": 0, "admin_min": 0.0, "info": 0, "info_min": 0.0,
        "prod": 6, "prod_min": 2.5,
        "bounce_pct": 3.0, "exit_pct": 8.0, "page_value": 0.0, "special": 0.0,
        "month": "มีนาคม", "visitor": "ผู้เข้าชมใหม่ (ครั้งแรก)", "weekend": WEEKDAY,
    },
}


# ----------------------------------------------------------------------------
# โหลดโมเดล / ค่าความแม่นยำ
# ----------------------------------------------------------------------------
def find_model_path() -> Path:
    """หาไฟล์โมเดล: .joblib ชื่อที่กำหนด -> .joblib ไฟล์อื่น -> แตกจากไฟล์ .zip"""
    if MODEL_PATH.exists():
        return MODEL_PATH
    candidates = sorted(BASE_DIR.glob("*.joblib"))
    if candidates:
        return candidates[0]

    for zip_path in sorted(BASE_DIR.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".joblib")]
            if not names:
                continue
            out_dir = Path(tempfile.gettempdir()) / "model_cache"
            out_dir.mkdir(exist_ok=True)
            target = out_dir / Path(names[0]).name
            if not target.exists():
                with zf.open(names[0]) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            return target

    files = ", ".join(sorted(p.name for p in BASE_DIR.iterdir())) or "(ว่าง)"
    raise FileNotFoundError(
        f"ไม่พบไฟล์โมเดล (.joblib หรือ .zip) ในโฟลเดอร์แอป — ไฟล์ที่มีอยู่: {files}"
    )


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model():
    return joblib.load(find_model_path())


def load_metrics() -> dict:
    """อ่านค่าความแม่นยำจาก metrics.json (ถ้าไม่มีจะคืนค่าว่าง)"""
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def as_percent(value):
    """แปลงค่า 0-1 (หรือ 0-100) เป็นข้อความเปอร์เซ็นต์"""
    if value is None:
        return None
    value = float(value)
    if value <= 1:
        value *= 100
    return f"{value:.2f}%"


# ----------------------------------------------------------------------------
# แปลงค่าจากฟอร์ม -> อินพุตของโมเดล
# ----------------------------------------------------------------------------
def to_model_inputs(v: dict) -> dict:
    """แปลงหน่วยบนหน้าจอ (นาที, %) เป็นหน่วยของข้อมูลจริง (วินาที, สัดส่วน)"""
    return {
        "Administrative": v["admin"],
        "Administrative_Duration": v["admin_min"] * 60,
        "Informational": v["info"],
        "Informational_Duration": v["info_min"] * 60,
        "ProductRelated": v["prod"],
        "ProductRelated_Duration": v["prod_min"] * 60,
        "BounceRates": v["bounce_pct"] / 100,
        "ExitRates": v["exit_pct"] / 100,
        "PageValues": v["page_value"],
        "SpecialDay": v["special"],
        "OperatingSystems": v["os"],
        "Browser": v["browser"],
        "Region": v["region"],
        "TrafficType": v["traffic"],
        "Weekend": v["weekend"] == WEEKEND,
        "Month": MONTHS[v["month"]],
        "VisitorType": VISITOR_TYPES[v["visitor"]],
    }


def build_features(data: dict, feature_cols) -> pd.DataFrame:
    """สร้างฟีเจอร์ 28 ตัวให้ตรงกับโมเดล (รวมการปรับสเกลแบบ MinMax)"""
    row = {c: 0.0 for c in feature_cols}

    # ฟีเจอร์ที่ปรับสเกล 0-1
    for key in [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues",
    ]:
        row[key] = data[key] / SCALE_MAX[key]

    total_pages = data["Administrative"] + data["Informational"] + data["ProductRelated"]
    total_duration = (
        data["Administrative_Duration"]
        + data["Informational_Duration"]
        + data["ProductRelated_Duration"]
    )
    row["TotalPages"] = total_pages / SCALE_MAX["TotalPages"]
    row["TotalDuration"] = total_duration / SCALE_MAX["TotalDuration"]

    # ฟีเจอร์ที่ไม่ปรับสเกล
    for key in ["SpecialDay", "OperatingSystems", "Browser", "Region", "TrafficType"]:
        row[key] = data[key]
    row["Weekend"] = int(data["Weekend"])

    month_col = f"Month_{data['Month']}"
    if month_col in row:
        row[month_col] = 1
    visitor_col = f"VisitorType_{data['VisitorType']}"
    if visitor_col in row:
        row[visitor_col] = 1

    return pd.DataFrame([row])[list(feature_cols)]


def describe_result(proba: float):
    """คืน (หัวข้อผล, คำแนะนำ, สี) ตามระดับความน่าจะเป็น"""
    if proba >= 0.5:
        return (
            "มีแนวโน้มที่จะซื้อสินค้า",
            "เป็นผู้เข้าชมที่น่าสนใจ ควรติดตามหรือเสนอโปรโมชันเพื่อช่วยปิดการขาย",
            "#2F5D50",
        )
    if proba >= 0.25:
        return (
            "ก้ำกึ่ง — อาจซื้อหรือไม่ซื้อก็ได้",
            "ลองกระตุ้นด้วยส่วนลดหรือการแจ้งเตือนสินค้าที่ค้างในตะกร้า",
            "#8A6D3B",
        )
    return (
        "มีแนวโน้มที่จะไม่ซื้อสินค้า",
        "ผู้เข้าชมยังไม่พร้อมซื้อ อาจแสดงสินค้าแนะนำเพื่อดึงความสนใจ",
        "#64748B",
    )


# ----------------------------------------------------------------------------
# สไตล์ (มินิมอล)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Noto Sans Thai', sans-serif; color: #1F2937; }
.stApp { background-color: #FAFAF9; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 880px; padding-top: 2.5rem; padding-bottom: 2rem; }

.app-header { text-align: center; padding-bottom: 1.25rem; border-bottom: 1px solid #E5E7EB; margin-bottom: 1.25rem; }
.app-title { font-size: 1.75rem; font-weight: 600; color: #1F2937; margin: 0; line-height: 1.4; }
.app-subtitle { font-size: 0.9rem; color: #6B7280; margin-top: 0.25rem; letter-spacing: 0.02em; }

.howto { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 0.75rem 1rem;
         font-size: 0.92rem; color: #374151; text-align: center; margin-bottom: 0.9rem; }
.howto b { color: #2F5D50; font-weight: 600; }

.section-label { font-size: 0.95rem; font-weight: 600; color: #1F2937; margin: 0.1rem 0 0; }
.section-hint { font-size: 0.82rem; color: #6B7280; margin: 0 0 0.75rem 0; }
.section-num { display: inline-block; width: 1.4rem; height: 1.4rem; line-height: 1.4rem; text-align: center;
               background: #2F5D50; color: #FFFFFF; border-radius: 50%; font-size: 0.78rem; margin-right: 0.45rem; }

[data-testid="stForm"] { border: none; padding: 0; background: transparent; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-color: #E5E7EB !important; border-radius: 10px; }

/* ปุ่มตัวอย่าง (นอกฟอร์ม): โปร่ง ขอบบาง */
.stButton > button {
    background-color: #FFFFFF; color: #2F5D50; border: 1px solid #D1D5DB; border-radius: 8px;
    padding: 0.45rem 0.75rem; font-weight: 500; width: 100%;
}
.stButton > button:hover { border-color: #2F5D50; color: #2F5D50; background-color: #F3F7F5; }

/* ปุ่มทำนายผล: เขียวเข้ม เต็มความกว้าง */
[data-testid="stFormSubmitButton"], [data-testid="stFormSubmitButton"] > div { width: 100% !important; }
[data-testid="stFormSubmitButton"] button {
    background-color: #2F5D50 !important; color: #FFFFFF !important; border: none !important;
    border-radius: 8px; padding: 0.7rem 1rem; font-weight: 500; font-size: 1rem; width: 100% !important;
}
[data-testid="stFormSubmitButton"] button:hover { background-color: #244a40 !important; }

.result-card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px;
               padding: 1.5rem; text-align: center; margin-top: 1.25rem; }
.result-label { font-size: 0.85rem; color: #6B7280; margin-bottom: 0.25rem; }
.result-value { font-size: 2.8rem; font-weight: 600; line-height: 1.2; }
.result-text { font-size: 1.1rem; font-weight: 500; margin: 0.25rem 0 1rem 0; }
.bar-bg { background: #F1F5F4; border-radius: 999px; height: 8px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }
.result-advice { font-size: 0.9rem; color: #4B5563; margin-top: 1rem; }
.result-note { font-size: 0.78rem; color: #9CA3AF; margin-top: 0.5rem; }

.metric-card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 1rem; text-align: center; }
.metric-card.primary { border-color: #2F5D50; }
.metric-name { font-size: 0.8rem; color: #6B7280; }
.metric-value { font-size: 1.6rem; font-weight: 600; color: #2F5D50; }

.app-footer { text-align: center; color: #6B7280; font-size: 0.85rem;
              border-top: 1px solid #E5E7EB; margin-top: 2.5rem; padding-top: 1rem; }
.app-footer b { color: #1F2937; font-weight: 500; }
</style>
"""


# ----------------------------------------------------------------------------
# ส่วนแสดงผล
# ----------------------------------------------------------------------------
def stretch(fn) -> dict:
    """ตัวเลือกให้ปุ่มกว้างเต็มพื้นที่ ที่ใช้ได้ทั้ง Streamlit เวอร์ชันใหม่และเก่า"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def apply_values(values: dict):
    """ใส่ค่าลงในฟอร์ม (ค่าที่ไม่ได้ระบุจะกลับเป็นค่าเริ่มต้น)"""
    for key, val in {**DEFAULTS, **values}.items():
        st.session_state[key] = val


def section(number: int, title: str, hint: str):
    st.markdown(
        f'<div class="section-label"><span class="section-num">{number}</span>{title}</div>'
        f'<div class="section-hint">{hint}</div>',
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        f"""
        <div class="app-header">
            <p class="app-title">{APP_TITLE}</p>
            <div class="app-subtitle">{APP_SUBTITLE}</div>
        </div>
        <div class="howto">
            วิธีใช้: <b>① เลือกตัวอย่าง</b> หรือกรอกข้อมูลเอง →
            <b>② กด “ทำนายผล”</b> → <b>③ อ่านผลด้านล่าง</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_presets():
    c1, c2, c3 = st.columns(3)
    c1.button("ตัวอย่าง: สนใจซื้อ", on_click=apply_values, args=(PRESETS["interested"],), **stretch(st.button))
    c2.button("ตัวอย่าง: แค่เข้ามาดู", on_click=apply_values, args=(PRESETS["browsing"],), **stretch(st.button))
    c3.button("ล้างค่า", on_click=apply_values, args=({},), **stretch(st.button))


def render_form():
    """แสดงฟอร์มรับข้อมูล คืนค่า (submitted, ค่าตามหน้าจอ)"""
    with st.form("predict_form"):
        with st.container(border=True):
            section(1, "การเข้าชมหน้าเว็บ", "ผู้ใช้เปิดดูกี่หน้า และใช้เวลานานแค่ไหน")
            c1, c2, c3 = st.columns(3)
            with c1:
                admin = st.number_input("หน้าบัญชีผู้ใช้ (หน้า)", 0, 100, step=1, key="admin",
                                        help="เช่น หน้าสมัครสมาชิก โปรไฟล์ ตั้งค่าบัญชี")
                admin_min = st.number_input("เวลาที่ใช้ (นาที)", 0.0, 2000.0, step=1.0, key="admin_min")
            with c2:
                info = st.number_input("หน้าข้อมูล/ช่วยเหลือ (หน้า)", 0, 100, step=1, key="info",
                                       help="เช่น หน้าติดต่อเรา เกี่ยวกับเรา วิธีสั่งซื้อ")
                info_min = st.number_input("เวลาที่ใช้ (นาที) ", 0.0, 2000.0, step=1.0, key="info_min")
            with c3:
                prod = st.number_input("หน้าสินค้า (หน้า)", 0, 1000, step=1, key="prod",
                                       help="จำนวนหน้าสินค้าที่ผู้ใช้เปิดดู")
                prod_min = st.number_input("เวลาที่ใช้ (นาที)  ", 0.0, 2000.0, step=1.0, key="prod_min")

        with st.container(border=True):
            section(2, "คุณภาพการเข้าชม", "ตัวเลขจากระบบวิเคราะห์เว็บไซต์ เช่น Google Analytics — ถ้าไม่ทราบให้ใช้ค่าเริ่มต้น")
            c1, c2, c3, c4 = st.columns(4)
            bounce = c1.number_input("ออกทันที (%)", 0.0, 100.0, step=0.5, format="%.1f", key="bounce_pct",
                                     help="สัดส่วนผู้ที่เข้ามาแล้วออกโดยไม่ทำอะไรต่อ (ส่วนใหญ่ไม่เกิน 20%)")
            exit_ = c2.number_input("ออกจากหน้า (%)", 0.0, 100.0, step=0.5, format="%.1f", key="exit_pct",
                                    help="สัดส่วนที่หน้าเว็บนั้นเป็นหน้าสุดท้ายก่อนผู้ใช้ออกจากเว็บ (ส่วนใหญ่ไม่เกิน 20%)")
            page_val = c3.number_input("มูลค่าหน้าเว็บ", 0.0, 500.0, step=1.0, key="page_value",
                                       help="Page Value: ค่าเฉลี่ยมูลค่าที่หน้าซึ่งผู้ใช้เปิดดูเคยสร้างก่อนเกิดการซื้อ "
                                            "ยิ่งสูงยิ่งใกล้ซื้อ (ไม่ทราบให้ใส่ 0) เป็นตัวแปรที่มีผลต่อการทำนายมากที่สุด")
            special = c4.select_slider(
                "ใกล้วันสำคัญ", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], key="special",
                format_func=lambda x: "ไม่ใกล้" if x == 0 else ("ตรงวัน" if x == 1 else f"{int(x * 100)}%"),
                help="ความใกล้วันสำคัญทางการค้า เช่น วาเลนไทน์ แม่ 0 = ไม่เกี่ยวข้อง 1 = ตรงกับวันนั้น",
            )

        with st.container(border=True):
            section(3, "ข้อมูลผู้เข้าชม", "ผู้ใช้เข้ามาเมื่อไหร่ และเป็นใคร")
            c1, c2, c3 = st.columns(3)
            month = c1.selectbox("เดือนที่เข้าชม", list(MONTHS.keys()), key="month")
            visitor = c2.selectbox("ประเภทผู้เข้าชม", list(VISITOR_TYPES.keys()), key="visitor")
            weekend = c3.radio("วันที่เข้าชม", [WEEKDAY, WEEKEND], key="weekend")

            with st.expander("ข้อมูลเทคนิคเพิ่มเติม (ไม่ต้องแก้ก็ได้)"):
                st.caption("เป็นรหัสตัวเลขตามชุดข้อมูลต้นแบบ ใช้ค่าเริ่มต้นได้")
                c1, c2, c3, c4 = st.columns(4)
                os_ = c1.number_input("ระบบปฏิบัติการ (1-8)", 1, 8, step=1, key="os")
                browser = c2.number_input("เบราว์เซอร์ (1-13)", 1, 13, step=1, key="browser")
                region = c3.number_input("ภูมิภาค (1-9)", 1, 9, step=1, key="region")
                traffic = c4.number_input("ช่องทางเข้าเว็บ (1-20)", 1, 20, step=1, key="traffic")

        submitted = st.form_submit_button("ทำนายผล", **stretch(st.form_submit_button))

    values = {
        "admin": admin, "admin_min": admin_min,
        "info": info, "info_min": info_min,
        "prod": prod, "prod_min": prod_min,
        "bounce_pct": bounce, "exit_pct": exit_,
        "page_value": page_val, "special": special,
        "month": month, "visitor": visitor, "weekend": weekend,
        "os": os_, "browser": browser, "region": region, "traffic": traffic,
    }
    return submitted, values


def render_result(model, values: dict):
    X = build_features(to_model_inputs(values), model.feature_names_in_)
    proba = float(model.predict_proba(X)[0][1])
    title, advice, color = describe_result(proba)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">โอกาสที่ผู้เข้าชมจะซื้อสินค้า</div>
            <div class="result-value" style="color:{color}">{proba * 100:.1f}%</div>
            <div class="result-text" style="color:{color}">{title}</div>
            <div class="bar-bg"><div class="bar-fill" style="width:{proba * 100:.1f}%; background:{color}"></div></div>
            <div class="result-advice">{advice}</div>
            <div class="result-note">ผลนี้เป็นการประมาณจากโมเดล ไม่ใช่ค่าที่แน่นอน</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics():
    st.markdown("---")
    section(4, "ความแม่นยำของระบบ", "ผลการประเมินโมเดลกับข้อมูลทดสอบที่โมเดลไม่เคยเห็นมาก่อน")

    metrics = load_metrics()
    items = [
        ("Accuracy", "ความแม่นยำ (Accuracy)", metrics.get("accuracy"), True),
        ("Precision", "Precision", metrics.get("precision"), False),
        ("Recall", "Recall", metrics.get("recall"), False),
        ("F1-score", "F1-score", metrics.get("f1_score"), False),
        ("ROC-AUC", "ROC-AUC", metrics.get("roc_auc"), False),
    ]
    items = [(n, th, as_percent(v), p) for n, th, v, p in items if v is not None]

    if not items:
        st.info("ยังไม่ได้ระบุค่าความแม่นยำ — กรอกค่าจากการประเมินโมเดลลงในไฟล์ metrics.json")
        return

    cols = st.columns(len(items))
    for col, (name, th, value, primary) in zip(cols, items):
        css = "metric-card primary" if primary else "metric-card"
        col.markdown(
            f'<div class="{css}"><div class="metric-name">{th}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    note = metrics.get("note")
    st.caption(note if note else "Accuracy คือสัดส่วนของการทำนายที่ถูกต้องทั้งหมด")


def render_footer():
    st.markdown(
        f'<div class="app-footer">ผู้พัฒนา<br><b>{DEVELOPERS}</b></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🛒", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    for key, val in DEFAULTS.items():
        st.session_state.setdefault(key, val)

    render_header()

    try:
        model = load_model()
    except Exception as e:  # noqa: BLE001
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        st.stop()

    render_presets()
    submitted, values = render_form()
    if submitted:
        render_result(model, values)

    render_metrics()
    render_footer()


if __name__ == "__main__":
    main()
