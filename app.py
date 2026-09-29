"""
Online Shoppers Purchase Intention — Streamlit App
โมเดล: Random Forest (random_forest_online_shoppers.joblib)
"""

import json
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
    "ผู้เข้าชมใหม่ (New)": "New_Visitor",
    "ผู้เข้าชมเดิม (Returning)": "Returning_Visitor",
    "อื่น ๆ (Other)": "Other",
}

# ----------------------------------------------------------------------------
# โหลดโมเดล / ค่าความแม่นยำ
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model():
    return joblib.load(MODEL_PATH)


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
# สร้างฟีเจอร์ให้ตรงกับโมเดล (28 ฟีเจอร์)
# ----------------------------------------------------------------------------
def build_features(data: dict, feature_cols) -> pd.DataFrame:
    row = {c: 0 for c in feature_cols}

    for key in [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        "OperatingSystems", "Browser", "Region", "TrafficType",
    ]:
        row[key] = data[key]

    row["Weekend"] = int(data["Weekend"])
    row["TotalPages"] = (
        data["Administrative"] + data["Informational"] + data["ProductRelated"]
    )
    row["TotalDuration"] = (
        data["Administrative_Duration"]
        + data["Informational_Duration"]
        + data["ProductRelated_Duration"]
    )

    month_col = f"Month_{data['Month']}"
    if month_col in row:
        row[month_col] = 1

    visitor_col = f"VisitorType_{data['VisitorType']}"
    if visitor_col in row:
        row[visitor_col] = 1

    return pd.DataFrame([row])[list(feature_cols)]


# ----------------------------------------------------------------------------
# สไตล์ (มินิมอล)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Noto Sans Thai', sans-serif;
    color: #1F2937;
}
.stApp { background-color: #FAFAF9; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 880px; padding-top: 2.5rem; padding-bottom: 2rem; }

.app-header { text-align: center; padding-bottom: 1.25rem; border-bottom: 1px solid #E5E7EB; margin-bottom: 1.75rem; }
.app-title { font-size: 1.75rem; font-weight: 600; color: #1F2937; margin: 0; line-height: 1.4; }
.app-subtitle { font-size: 0.9rem; color: #6B7280; margin-top: 0.25rem; letter-spacing: 0.02em; }

.section-label { font-size: 0.8rem; font-weight: 500; color: #6B7280; letter-spacing: 0.06em;
                 text-transform: uppercase; margin: 0.25rem 0 0.75rem 0; }

[data-testid="stForm"] { border: none; padding: 0; background: transparent; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-color: #E5E7EB !important; border-radius: 10px; }

.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background-color: #2F5D50; color: #FFFFFF; border: none; border-radius: 8px;
    padding: 0.6rem 1rem; font-weight: 500; width: 100%; transition: background-color .15s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background-color: #244a40; color: #FFFFFF; border: none;
}

.result-card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px;
               padding: 1.5rem; text-align: center; margin-top: 1.25rem; }
.result-label { font-size: 0.85rem; color: #6B7280; margin-bottom: 0.25rem; }
.result-value { font-size: 2.6rem; font-weight: 600; line-height: 1.2; }
.result-text { font-size: 1.05rem; font-weight: 500; margin: 0.25rem 0 1rem 0; }
.bar-bg { background: #F1F5F4; border-radius: 999px; height: 8px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }

.metric-card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px;
               padding: 1rem; text-align: center; }
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
def render_header():
    st.markdown(
        f"""
        <div class="app-header">
            <p class="app-title">{APP_TITLE}</p>
            <div class="app-subtitle">{APP_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_form():
    """แสดงฟอร์มรับข้อมูล คืนค่า (submitted, data)"""
    with st.form("predict_form"):
        with st.container(border=True):
            st.markdown('<div class="section-label">พฤติกรรมการเข้าชมหน้าเว็บ</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                admin = st.number_input("หน้าบัญชี/ตั้งค่า (จำนวนหน้า)", 0, 100, 0)
                admin_d = st.number_input("เวลาที่ใช้ (วินาที)", 0.0, 10000.0, 0.0, 10.0, key="ad")
            with c2:
                info = st.number_input("หน้าข้อมูลทั่วไป (จำนวนหน้า)", 0, 100, 0)
                info_d = st.number_input("เวลาที่ใช้ (วินาที)", 0.0, 10000.0, 0.0, 10.0, key="id")
            with c3:
                prod = st.number_input("หน้าสินค้า (จำนวนหน้า)", 0, 1000, 10)
                prod_d = st.number_input("เวลาที่ใช้ (วินาที)", 0.0, 100000.0, 300.0, 10.0, key="pd")

        with st.container(border=True):
            st.markdown('<div class="section-label">ตัวชี้วัดของเว็บไซต์</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            bounce = c1.number_input("Bounce Rate", 0.0, 1.0, 0.02, 0.01, format="%.3f")
            exit_ = c2.number_input("Exit Rate", 0.0, 1.0, 0.04, 0.01, format="%.3f")
            page_val = c3.number_input("Page Value", 0.0, 500.0, 0.0, 1.0)
            special = c4.select_slider("Special Day", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0)

        with st.container(border=True):
            st.markdown('<div class="section-label">ข้อมูลผู้เข้าชม</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            month = c1.selectbox("เดือน", list(MONTHS.keys()), index=list(MONTHS).index("พฤศจิกายน"))
            visitor = c2.selectbox("ประเภทผู้เข้าชม", list(VISITOR_TYPES.keys()), index=1)
            weekend = c3.selectbox("วันที่เข้าชม", ["วันธรรมดา", "วันหยุดสุดสัปดาห์"]) == "วันหยุดสุดสัปดาห์"

            c1, c2, c3, c4 = st.columns(4)
            os_ = c1.number_input("Operating System", 1, 8, 2)
            browser = c2.number_input("Browser", 1, 13, 2)
            region = c3.number_input("Region", 1, 9, 1)
            traffic = c4.number_input("Traffic Type", 1, 20, 2)

        submitted = st.form_submit_button("ทำนายผล")

    data = {
        "Administrative": admin, "Administrative_Duration": admin_d,
        "Informational": info, "Informational_Duration": info_d,
        "ProductRelated": prod, "ProductRelated_Duration": prod_d,
        "BounceRates": bounce, "ExitRates": exit_,
        "PageValues": page_val, "SpecialDay": special,
        "OperatingSystems": os_, "Browser": browser,
        "Region": region, "TrafficType": traffic,
        "Weekend": weekend,
        "Month": MONTHS[month],
        "VisitorType": VISITOR_TYPES[visitor],
    }
    return submitted, data


def render_result(model, data: dict):
    X = build_features(data, model.feature_names_in_)
    proba = float(model.predict_proba(X)[0][1])
    will_buy = int(model.predict(X)[0]) == 1

    color = "#2F5D50" if will_buy else "#64748B"
    text = "มีแนวโน้มที่จะซื้อสินค้า" if will_buy else "มีแนวโน้มที่จะไม่ซื้อสินค้า"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">โอกาสที่จะซื้อสินค้า</div>
            <div class="result-value" style="color:{color}">{proba * 100:.1f}%</div>
            <div class="result-text" style="color:{color}">{text}</div>
            <div class="bar-bg"><div class="bar-fill" style="width:{proba * 100:.1f}%; background:{color}"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics():
    st.markdown("---")
    st.markdown('<div class="section-label">ความแม่นยำของระบบ</div>', unsafe_allow_html=True)

    metrics = load_metrics()
    items = [
        ("Accuracy", "ความแม่นยำ", metrics.get("accuracy"), True),
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
    if note:
        st.caption(note)


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

    render_header()

    try:
        model = load_model()
    except Exception as e:  # noqa: BLE001
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        st.stop()

    submitted, data = render_form()
    if submitted:
        render_result(model, data)

    render_metrics()
    render_footer()


if __name__ == "__main__":
    main()
