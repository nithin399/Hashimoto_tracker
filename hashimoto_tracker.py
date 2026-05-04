import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import os
import pandas as pd
import plotly.express as px
from datetime import datetime, date
import gspread
from google.oauth2.service_account import Credentials

# ═════════════════════════════════════════════════════
# CONFIGURATION — reads Streamlit secrets first
# ═════════════════════════════════════════════════════
def get_secret(key, default=""):
    """Read from st.secrets first, then environment variables."""
    try:
        val = st.secrets.get(key, None)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, default)

GROQ_API_KEY   = get_secret("GROQ_API_KEY")
SPREADSHEET_ID = get_secret("SPREADSHEET_ID")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SHEET_NAMES = {
    "logs":        "Daily Logs",
    "labs":        "Lab Results",
    "supplements": "Supplements",
    "flares":      "Flare History",
    "meds":        "Medication"
}

# ═════════════════════════════════════════════════════
# GOOGLE SHEETS HELPERS
# ═════════════════════════════════════════════════════
@st.cache_resource
def get_gsheet_client():
    """Authenticate with Google Sheets — handles all private_key formats."""
    try:
        creds_dict = None

        # Method 1: st.secrets as TOML section [GOOGLE_CREDENTIALS]
        try:
            raw = dict(st.secrets["GOOGLE_CREDENTIALS"])
            # Fix private_key: replace literal \n with real newlines
            if "private_key" in raw:
                raw["private_key"] = raw["private_key"].replace("\\n", "\n").replace("\n", "\n")
            creds_dict = raw
        except Exception:
            pass

        # Method 2: st.secrets as raw JSON string GOOGLE_CREDENTIALS_JSON
        if not creds_dict:
            try:
                raw_json = st.secrets.get("GOOGLE_CREDENTIALS_JSON", "")
                if raw_json:
                    creds_dict = json.loads(raw_json)
            except Exception:
                pass

        # Method 3: environment variable as JSON string (local dev)
        if not creds_dict:
            raw_json = os.environ.get("GOOGLE_CREDENTIALS", "")
            if raw_json:
                creds_dict = json.loads(raw_json)

        if not creds_dict:
            return None

        # Fix private_key newlines if needed
        if "private_key" in creds_dict:
            pk = creds_dict["private_key"]
            if "\\n" in pk:
                pk = pk.replace("\\n", "\n")
            creds_dict["private_key"] = pk

        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return gspread.authorize(creds)

    except Exception as e:
        st.error(f"Google Sheets connection error: {e}")
        return None

def get_or_create_sheet(client, sheet_name):
    """Get existing sheet or create it."""
    try:
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        try:
            return spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            return spreadsheet.add_worksheet(title=sheet_name, rows=10000, cols=50)
    except Exception as e:
        st.error(f"Sheet error: {e}")
        return None

@st.cache_data(ttl=60)
def sheet_to_df(_client, sheet_name):
    """Read a sheet into a DataFrame. Cached for 60 seconds to avoid quota."""
    try:
        ws = get_or_create_sheet(_client, sheet_name)
        if ws is None:
            return pd.DataFrame()
        data = ws.get_all_records()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def append_row(client, sheet_name, row_dict):
    """Append a row to a sheet. Creates header if sheet is empty."""
    try:
        ws = get_or_create_sheet(client, sheet_name)
        if ws is None:
            return False
        existing = ws.get_all_values()
        if not existing:
            # Write header
            ws.append_row(list(row_dict.keys()))
        ws.append_row(list(row_dict.values()))
        return True
    except Exception as e:
        st.error(f"Save error: {e}")
        return False

def upsert_row(client, sheet_name, key_col, key_val, row_dict):
    """Update row if key exists, else append. Used for daily logs (one per day)."""
    try:
        ws = get_or_create_sheet(client, sheet_name)
        if ws is None:
            return False
        existing = ws.get_all_values()
        if not existing:
            ws.append_row(list(row_dict.keys()))
            ws.append_row(list(row_dict.values()))
            return True
        headers = existing[0]
        try:
            key_idx = headers.index(key_col) + 1  # 1-indexed
        except ValueError:
            ws.append_row(list(row_dict.keys()))
            ws.append_row(list(row_dict.values()))
            return True
        # Search for existing row
        col_values = ws.col_values(key_idx)
        if key_val in col_values:
            row_num = col_values.index(key_val) + 1
            # Build full row in header order
            new_row = [str(row_dict.get(h, "")) for h in headers]
            ws.update(f"A{row_num}", [new_row])
        else:
            # Ensure row matches header order
            new_row = [str(row_dict.get(h, "")) for h in headers]
            ws.append_row(new_row)
        return True
    except Exception as e:
        st.error(f"Upsert error: {e}")
        return False

def get_row_by_key(client, sheet_name, key_col, key_val):
    """Get a single row as dict by key value."""
    try:
        ws = get_or_create_sheet(client, sheet_name)
        if ws is None:
            return {}
        df = pd.DataFrame(ws.get_all_records())
        if df.empty or key_col not in df.columns:
            return {}
        match = df[df[key_col] == key_val]
        if match.empty:
            return {}
        return match.iloc[-1].to_dict()
    except Exception:
        return {}

# ═════════════════════════════════════════════════════
# LOCAL FALLBACK STORAGE (when no Google Sheets)
# ═════════════════════════════════════════════════════
LOGS_FILE        = "hashimoto_logs.json"
LABS_FILE        = "hashimoto_labs.json"
MEDS_FILE        = "hashimoto_meds.json"
SUPPLEMENTS_FILE = "hashimoto_supplements.json"
FLARES_FILE      = "hashimoto_flares.json"

def load_json(fp):
    if not os.path.exists(fp):
        return {}
    with open(fp) as f:
        return json.load(f)

def save_json(fp, data):
    with open(fp, "w") as f:
        json.dump(data, f, indent=2)

# ═════════════════════════════════════════════════════
# UNIFIED SAVE / LOAD (Sheets if connected, JSON fallback)
# ═════════════════════════════════════════════════════
def save_daily_log(data, client=None):
    data["date"] = str(date.today())
    if client and SPREADSHEET_ID:
        upsert_row(client, SHEET_NAMES["logs"], "date", data["date"], data)
    # Always save locally too as backup
    logs = load_json(LOGS_FILE)
    logs[data["date"]] = data
    save_json(LOGS_FILE, logs)

@st.cache_data(ttl=60)
def load_daily_logs(_client=None):
    if _client and SPREADSHEET_ID:
        try:
            df = sheet_to_df(_client, SHEET_NAMES["logs"])
            if not df.empty and "date" in df.columns:
                return df.set_index("date").to_dict(orient="index")
        except Exception:
            pass
    return load_json(LOGS_FILE)

def save_lab(data, client=None):
    data["timestamp"] = str(datetime.now())
    if client and SPREADSHEET_ID:
        append_row(client, SHEET_NAMES["labs"], data)
    labs = load_json(LABS_FILE)
    labs[data["timestamp"]] = data
    save_json(LABS_FILE, labs)

@st.cache_data(ttl=60)
def load_labs(_client=None):
    if _client and SPREADSHEET_ID:
        try:
            df = sheet_to_df(_client, SHEET_NAMES["labs"])
            if not df.empty:
                return df.to_dict(orient="records")
        except Exception:
            pass
    labs = load_json(LABS_FILE)
    return list(labs.values())

def save_supplement(supps, client=None):
    save_json(SUPPLEMENTS_FILE, supps)
    if client and SPREADSHEET_ID:
        try:
            ws = get_or_create_sheet(client, SHEET_NAMES["supplements"])
            if ws:
                ws.clear()
                rows = list(supps.values())
                if rows:
                    ws.append_row(list(rows[0].keys()))
                    for r in rows:
                        ws.append_row(list(r.values()))
        except Exception:
            pass

@st.cache_data(ttl=60)
def load_supplements(_client=None):
    if _client and SPREADSHEET_ID:
        try:
            df = sheet_to_df(_client, SHEET_NAMES["supplements"])
            if not df.empty and "name" in df.columns:
                result = {}
                for _, row in df.iterrows():
                    d = row.to_dict()
                    sid = d.get("name", "").lower().replace(" ", "_")
                    result[sid] = d
                return result
        except Exception:
            pass
    return load_json(SUPPLEMENTS_FILE)

def save_flare_record(flare_data, client=None):
    flares = load_json(FLARES_FILE)
    ts = str(datetime.now())
    flares[ts] = flare_data
    save_json(FLARES_FILE, flares)
    if client and SPREADSHEET_ID:
        flat = {
            "timestamp": ts,
            "date": flare_data["date"],
            "alerts": " | ".join(flare_data["alerts"]),
        }
        flat.update(flare_data.get("snapshot", {}))
        append_row(client, SHEET_NAMES["flares"], flat)

@st.cache_data(ttl=60)
def load_flares(_client=None):
    if _client and SPREADSHEET_ID:
        try:
            df = sheet_to_df(_client, SHEET_NAMES["flares"])
            if not df.empty:
                result = {}
                for _, row in df.iterrows():
                    d = row.to_dict()
                    ts = d.get("timestamp", str(datetime.now()))
                    alerts = d.get("alerts", "").split(" | ") if d.get("alerts") else []
                    result[ts] = {"date": d.get("date",""), "alerts": alerts, "snapshot": d}
                return result
        except Exception:
            pass
    return load_json(FLARES_FILE)

# ═════════════════════════════════════════════════════
# FLARE DETECTION
# ═════════════════════════════════════════════════════
FLARE_THRESHOLDS = {
    "energy":        ("low",  4),
    "brain_fog":     ("high", 7),
    "joint_pain":    ("high", 7),
    "carpal_tunnel": ("high", 6),
    "mood":          ("low",  4),
    "stress_level":  ("high", 8),
}

def detect_flare(log):
    alerts = []
    for key, (direction, threshold) in FLARE_THRESHOLDS.items():
        val = log.get(key)
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        if direction == "low" and val <= threshold:
            alerts.append(f"**{key.replace('_',' ').title()}** is very low ({val:.0f}/10)")
        if direction == "high" and val >= threshold:
            alerts.append(f"**{key.replace('_',' ').title()}** is very high ({val:.0f}/10)")
    return alerts

# ═════════════════════════════════════════════════════
# AI HELPERS
# ═════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a knowledgeable Hashimoto's thyroiditis health coach.
You help patients track symptoms, understand their condition,
manage lifestyle, and prepare questions for their doctor.

Important rules:
- Always remind user you are not a doctor
- Never change medication advice, always refer to their doctor
- Be empathetic, Hashimoto's is exhausting and invisible
- Give practical, evidence-based lifestyle tips
- Help interpret symptoms in context of Hashimoto's
- Suggest questions they can ask their endocrinologist

You know about TSH, Free T3, Free T4, Total T3, Total T4, TPO antibodies,
TG Antibodies, TgAb, RA Factor, Levothyroxine, NDT, T3/T4 combo therapy,
gluten-thyroid connection, AIP diet, anti-inflammatory foods, histamine intolerance,
adrenal fatigue, cortisol, stress impact,
sleep, exercise, supplements like Selenium, Vitamin D, Vitamin B12, Magnesium,
brain fog, fatigue, weight gain, hair loss, cold intolerance,
menstrual irregularities, constipation, soy and food sensitivities."""

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.7
    )

def chat(messages, user_input):
    llm = get_llm()
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in messages:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=user_input))
    return llm.invoke(msgs).content

def ask(prompt_text):
    llm = get_llm()
    msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt_text)]
    return llm.invoke(msgs).content

# ═════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════
st.set_page_config(page_title="Hashimoto Tracker", page_icon="🦋", layout="wide")

st.markdown("""
<style>
  .flare-box {
    background: #fff3f3;
    border-left: 4px solid #e07b7b;
    padding: 0.8rem 1.2rem;
    border-radius: 6px;
    margin-bottom: 0.5rem;
  }
  .ok-box {
    background: #f0fff4;
    border-left: 4px solid #4fa3a0;
    padding: 0.8rem 1.2rem;
    border-radius: 6px;
  }
  .warn-box {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    padding: 0.8rem 1.2rem;
    border-radius: 6px;
    margin-bottom: 1rem;
  }
  .connected-box {
    background: #e8f5e9;
    border-left: 4px solid #43a047;
    padding: 0.6rem 1rem;
    border-radius: 6px;
    margin-bottom: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)

st.title("🦋 Hashimoto Tracker & Coach")
st.caption("Private · Secure · Cloud-backed · Your personal health companion")

# ═════════════════════════════════════════════════════
# SIDEBAR — Settings & Connection Status
# ═════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")

    # ── Groq API ──────────────────────────────────────
    # Re-read in case secrets loaded after module start
    GROQ_API_KEY   = get_secret("GROQ_API_KEY")
    SPREADSHEET_ID = get_secret("SPREADSHEET_ID")

    if GROQ_API_KEY:
        st.markdown('<div class="connected-box">✅ AI Connected (Groq)</div>',
                    unsafe_allow_html=True)
    else:
        user_key = st.text_input("Groq API Key", type="password",
                                  placeholder="gsk_...",
                                  help="Get free key at console.groq.com")
        if user_key:
            GROQ_API_KEY = user_key
            os.environ["GROQ_API_KEY"] = user_key
            st.rerun()
        st.caption("Get free key → [console.groq.com](https://console.groq.com)")

    st.divider()

    # ── Google Sheets ─────────────────────────────────
    st.subheader("☁️ Google Sheets Storage")
    if SPREADSHEET_ID:
        st.markdown('<div class="connected-box">✅ Google Sheets Connected</div>',
                    unsafe_allow_html=True)
    else:
        sid = st.text_input("Spreadsheet ID",
                             placeholder="Paste your Sheet ID here",
                             help="From your Google Sheet URL")
        if sid:
            SPREADSHEET_ID = sid
            os.environ["SPREADSHEET_ID"] = sid
            st.rerun()
        st.caption("See SETUP_GUIDE.md for instructions")

    # Init Google client
    client = get_gsheet_client() if SPREADSHEET_ID else None

    if client:
        st.markdown('<div class="connected-box">☁️ Syncing to Google Sheets</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">💾 Local storage only</div>',
                    unsafe_allow_html=True)
        client = None

    st.divider()
    st.caption("🦋 Hashimoto Tracker v2.0")

# ═════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Daily Log",
    "🧪 Lab Results",
    "📊 Trends",
    "💬 AI Coach",
    "💊 Supplements",
    "🚨 Flare Alerts",
    "📄 Doctor Report",
])

# ═════════════════════════════════════════════════════
# TAB 1 — Daily Log
# ═════════════════════════════════════════════════════
with tab1:
    st.header("📋 How Are You Feeling Today?")
    today    = str(date.today())
    all_logs = load_daily_logs(client)
    existing = all_logs.get(today, {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("😴 Energy & Mood")
        energy        = st.slider("Energy level", 1, 10, int(existing.get("energy", 5)))
        mood          = st.slider("Mood", 1, 10, int(existing.get("mood", 5)))
        brain_fog     = st.slider("Brain fog (1=none, 10=severe)", 1, 10, int(existing.get("brain_fog", 1)))
        sleep_hrs     = st.number_input("Hours slept", 0.0, 12.0, float(existing.get("sleep_hrs", 7.0)), 0.5)
        sleep_quality = st.slider("Sleep quality", 1, 10, int(existing.get("sleep_quality", 5)))
        bed_time      = st.time_input("🌙 Off to Bed time",
                            datetime.strptime(str(existing.get("bed_time", "22:30"))[:5], "%H:%M").time())
        wake_time     = st.time_input("☀️ Up from Bed time",
                            datetime.strptime(str(existing.get("wake_time", "07:00"))[:5], "%H:%M").time())

    with col2:
        st.subheader("🌡️ Physical Symptoms")
        weight           = st.number_input("Weight (kg)", 30.0, 200.0, float(existing.get("weight", 65.0)), 0.1)
        temp_opts        = ["Normal", "Cold", "Hot", "Both"]
        temp_sensitivity = st.selectbox("Temperature sensitivity", temp_opts,
                               index=temp_opts.index(existing.get("temp_sensitivity", "Normal")))
        hair_opts = ["None", "Mild", "Moderate", "Severe"]
        hair_loss = st.selectbox("Hair loss today", hair_opts,
                        index=hair_opts.index(existing.get("hair_loss", "None")))
        joint_pain        = st.slider("Joint/muscle pain (1=none)", 1, 10, int(existing.get("joint_pain", 1)))
        carpal_tunnel     = st.slider("Carpal tunnel pain (1=none)", 1, 10, int(existing.get("carpal_tunnel", 1)))
        bloating          = st.slider("Bloating/digestion (1=none)", 1, 10, int(existing.get("bloating", 1)))
        constipation_opts = ["None", "Mild", "Moderate", "Severe"]
        constipation      = st.selectbox("Constipation today", constipation_opts,
                               index=constipation_opts.index(existing.get("constipation", "None")))
        menstrual_notes   = st.text_area("🌸 Menstrual Cycle Notes",
                               str(existing.get("menstrual_notes", "")),
                               placeholder="e.g. Day 3, heavy flow, cramps, spotting...")

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("💊 Medication")
        meds_data = load_json(MEDS_FILE)
        med_name  = st.text_input("Medication name", meds_data.get("name", "Levothyroxine"))
        med_dose  = st.text_input("Dose", meds_data.get("dose", "50mcg"))
        med_taken = st.checkbox("Taken today (fasting)?", bool(existing.get("med_taken", False)))
        med_time  = st.time_input("Time taken",
                       datetime.strptime(str(existing.get("med_time", "07:00"))[:5], "%H:%M").time())
        save_json(MEDS_FILE, {"name": med_name, "dose": med_dose})

    with col4:
        st.subheader("🍽️ Food & Lifestyle")
        gluten_free      = st.checkbox("Gluten free today?",             bool(existing.get("gluten_free", False)))
        dairy_free       = st.checkbox("Dairy free today?",              bool(existing.get("dairy_free", False)))
        soy_free         = st.checkbox("Soy free today?",                bool(existing.get("soy_free", False)))
        histamine_free   = st.checkbox("Histamine friendly diet today?",  bool(existing.get("histamine_free", False)))
        no_allergy_foods = st.checkbox("No beans & allergy foods today?", bool(existing.get("no_allergy_foods", False)))
        hydration        = st.number_input("💧 Hydration (glasses of water)", 0, 20, int(existing.get("hydration", 8)))
        stress_level     = st.slider("Stress level", 1, 10, int(existing.get("stress_level", 5)))
        notes            = st.text_area("Notes / symptoms", str(existing.get("notes", "")))

        st.subheader("🏃 Exercise Today")
        ex_walking = st.checkbox("🚶 Walking", bool(existing.get("ex_walking", False)))
        ex_weights = st.checkbox("🏋️ Weights", bool(existing.get("ex_weights", False)))
        ex_yoga    = st.checkbox("🧘 Yoga",    bool(existing.get("ex_yoga", False)))
        ex_other   = st.text_input("Other exercise", str(existing.get("ex_other", "")))

    if st.button("✅ Save Today's Log", use_container_width=True):
        new_log = dict(
            date=today,
            energy=energy, mood=mood, brain_fog=brain_fog,
            sleep_hrs=sleep_hrs, sleep_quality=sleep_quality,
            bed_time=str(bed_time), wake_time=str(wake_time),
            weight=weight, temp_sensitivity=temp_sensitivity,
            hair_loss=hair_loss, joint_pain=joint_pain,
            bloating=bloating, carpal_tunnel=carpal_tunnel, constipation=constipation,
            menstrual_notes=menstrual_notes,
            med_taken=med_taken, med_time=str(med_time),
            gluten_free=gluten_free, dairy_free=dairy_free,
            soy_free=soy_free, histamine_free=histamine_free,
            no_allergy_foods=no_allergy_foods,
            hydration=hydration, stress_level=stress_level,
            ex_walking=ex_walking, ex_weights=ex_weights,
            ex_yoga=ex_yoga, ex_other=ex_other,
            notes=notes
        )
        with st.spinner("Saving..."):
            save_daily_log(new_log, client)
            st.cache_data.clear()
        alerts = detect_flare(new_log)
        if alerts:
            flare_data = {
                "date": today,
                "alerts": alerts,
                "snapshot": {k: new_log.get(k) for k in FLARE_THRESHOLDS}
            }
            save_flare_record(flare_data, client)
            st.warning("🚨 Possible flare detected! Check the Flare Alerts tab.")
            for a in alerts:
                st.markdown(f'<div class="flare-box">⚠️ {a}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-box">✅ Log saved — no flare indicators today!</div>',
                        unsafe_allow_html=True)
        st.balloons()

# ═════════════════════════════════════════════════════
# TAB 2 — Lab Results
# ═════════════════════════════════════════════════════
with tab2:
    st.header("🧪 Lab Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Thyroid Panel")
        tsh      = st.number_input("TSH (mIU/L)",        0.0,  20.0,   2.5,  0.01)
        free_t4  = st.number_input("Free T4 (ng/dL)",    0.0,   5.0,   1.3,  0.01)
        free_t3  = st.number_input("Free T3 (pg/mL)",    0.0,  10.0,   3.5,  0.01)
        total_t3 = st.number_input("Total T3 (ng/mL)",   0.0, 400.0, 120.0,  0.1)
        total_t4 = st.number_input("Total T4 (mg/dL)",   0.0,  20.0,   8.0,  0.1)

    with col2:
        st.subheader("Antibodies & Other")
        tpo_ab    = st.number_input("TPO Antibodies (IU/mL)",  0.0, 5000.0,  1.0, 0.1)
        tg_ab     = st.number_input("TG Antibodies (IU/mL)",   0.0, 5000.0,  1.0, 0.1)
        ra_factor = st.number_input("RA Factor (IU/mL)",       0.0,  500.0, 10.0, 0.1)
        vit_d     = st.number_input("Vitamin D (ng/mL)",       0.0,  200.0, 30.0, 0.1)
        vit_b12   = st.number_input("Vitamin B12 (pg/mL)",     0.0, 2000.0,300.0, 1.0)
        ferritin  = st.number_input("Ferritin (ng/mL)",        0.0,  500.0, 70.0, 0.1)

    lab_notes = st.text_area("Lab notes / how you felt")
    lab_date  = st.date_input("Lab date", date.today())

    if st.button("💾 Save Lab Results", use_container_width=True):
        lab_data = dict(
            date=str(lab_date), tsh=tsh,
            free_t4=free_t4, free_t3=free_t3,
            total_t3=total_t3, total_t4=total_t4,
            tpo_ab=tpo_ab, tg_ab=tg_ab,
            ra_factor=ra_factor, vit_d=vit_d,
            vit_b12=vit_b12, ferritin=ferritin,
            notes=lab_notes
        )
        with st.spinner("Saving lab results..."):
            save_lab(lab_data, client)
            st.cache_data.clear()
        st.success("✅ Lab results saved!")

    st.divider()
    st.subheader("📊 Reference Ranges")
    ref = pd.DataFrame({
        "Marker": [
            "TSH", "Free T4", "Free T3", "Total T3", "Total T4",
            "TPO Antibodies", "TG Antibodies", "RA Factor",
            "Vitamin D", "Vitamin B12", "Ferritin"
        ],
        "Optimal (Hashimoto's)": [
            "0.5-2 mIU/L", "1.16-1.78 ng/dL", "3.25-4.5 pg/mL",
            "100-180 ng/mL", "6-12 mg/dL",
            "<2 IU/mL", "<2 IU/mL", "<15 IU/mL",
            "30-100 ng/mL", "211-911 pg/mL", "10-291 ng/mL"
        ],
        "Standard Lab Range": [
            "0.4-4.0 mIU/L", "0.93-1.7 ng/dL", "2.0-4.4 pg/mL",
            "80-200 ng/mL", "5.1-14.1 mg/dL",
            "<34 IU/mL", "<115 IU/mL", "<14 IU/mL",
            "20-50 ng/mL", "160-950 pg/mL", "12-300 ng/mL"
        ],
        "What It Means": [
            "Thyroid stimulating hormone — lower is better for Hashi",
            "Active thyroid hormone — aim higher in range",
            "Active T3 — brain, energy, metabolism",
            "Overall T3 pool",
            "Overall T4 pool",
            "Thyroid peroxidase antibodies — lower = less attack",
            "Thyroglobulin antibodies — lower = less attack",
            "Rheumatoid factor — checks autoimmune activity",
            "Critical for immune & thyroid function",
            "Energy, nerve function — often low in Hashi",
            "Iron stores — low = fatigue & hair loss"
        ]
    })
    st.dataframe(ref, width="stretch")

    # Show history
    st.divider()
    st.subheader("📋 Lab History")
    lab_history = load_labs(client)
    if lab_history:
        hist_df = pd.DataFrame(lab_history)
        st.dataframe(hist_df, width="stretch")
    else:
        st.info("No lab results saved yet.")

# ═════════════════════════════════════════════════════
# TAB 3 — Trends
# ═════════════════════════════════════════════════════
with tab3:
    st.header("📊 Your Trends")
    all_logs = load_daily_logs(client)

    if len(all_logs) < 2:
        st.info("Log at least 2 days to see trends.")
    else:
        df = pd.DataFrame(all_logs).T.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"])
        num_cols = ["energy","mood","brain_fog","sleep_hrs","weight",
                    "stress_level","joint_pain","carpal_tunnel","bloating","hydration"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        col1, col2 = st.columns(2)
        with col1:
            if all(c in df.columns for c in ["energy","mood","brain_fog"]):
                fig = px.line(df, x="date", y=["energy","mood","brain_fog"],
                              title="Energy, Mood & Brain Fog",
                              color_discrete_map={
                                  "energy":"#4fa3a0",
                                  "mood":"#3498db",
                                  "brain_fog":"#e07b7b"})
                st.plotly_chart(fig, use_container_width=True)
            if "sleep_hrs" in df.columns:
                fig2 = px.bar(df, x="date", y="sleep_hrs", title="Sleep Hours",
                              color="sleep_hrs", color_continuous_scale="Teal")
                st.plotly_chart(fig2, use_container_width=True)
            if "hydration" in df.columns:
                fig5 = px.bar(df, x="date", y="hydration",
                              title="Daily Hydration (glasses)",
                              color="hydration", color_continuous_scale="Blues")
                st.plotly_chart(fig5, use_container_width=True)
        with col2:
            if "weight" in df.columns:
                fig3 = px.line(df, x="date", y="weight",
                               title="Weight (kg)", line_shape="spline")
                fig3.update_traces(line_color="#9b59b6")
                st.plotly_chart(fig3, use_container_width=True)
            if all(c in df.columns for c in ["stress_level","energy"]):
                fig4 = px.scatter(df, x="stress_level", y="energy",
                                  title="Stress vs Energy", trendline="ols",
                                  color="mood", color_continuous_scale="RdYlGn")
                st.plotly_chart(fig4, use_container_width=True)

        # Lab trends
        lab_history = load_labs(client)
        if len(lab_history) >= 2:
            st.divider()
            st.subheader("🧪 Lab Trends")
            lab_df = pd.DataFrame(lab_history)
            lab_df["date"] = pd.to_datetime(lab_df["date"])
            for col in ["tsh","free_t4","free_t3","tpo_ab","tg_ab","vit_d","vit_b12","ferritin"]:
                if col in lab_df.columns:
                    lab_df[col] = pd.to_numeric(lab_df[col], errors="coerce")
            col1, col2 = st.columns(2)
            with col1:
                if "tsh" in lab_df.columns:
                    fig_tsh = px.line(lab_df, x="date", y="tsh", title="TSH Over Time")
                    fig_tsh.add_hline(y=0.5, line_dash="dash", annotation_text="Optimal low")
                    fig_tsh.add_hline(y=2.0, line_dash="dash", annotation_text="Optimal high")
                    st.plotly_chart(fig_tsh, use_container_width=True)
                if "tpo_ab" in lab_df.columns:
                    fig_tpo = px.line(lab_df, x="date", y="tpo_ab",
                                      title="TPO Antibodies Over Time")
                    fig_tpo.add_hline(y=2, line_dash="dash",
                                      annotation_text="Optimal <2")
                    st.plotly_chart(fig_tpo, use_container_width=True)
            with col2:
                if "vit_d" in lab_df.columns:
                    fig_vd = px.line(lab_df, x="date", y="vit_d", title="Vitamin D Over Time")
                    fig_vd.add_hline(y=30, line_dash="dash", annotation_text="Min optimal")
                    fig_vd.add_hline(y=100, line_dash="dash", annotation_text="Max optimal")
                    st.plotly_chart(fig_vd, use_container_width=True)
                if "vit_b12" in lab_df.columns:
                    fig_b12 = px.line(lab_df, x="date", y="vit_b12",
                                      title="Vitamin B12 Over Time")
                    fig_b12.add_hline(y=211, line_dash="dash", annotation_text="Optimal low")
                    st.plotly_chart(fig_b12, use_container_width=True)

# ═════════════════════════════════════════════════════
# TAB 4 — AI Coach
# ═════════════════════════════════════════════════════
with tab4:
    st.header("💬 Hashimoto Coach")
    st.caption("Ask anything about your condition — powered by Groq AI")

    if not GROQ_API_KEY:
        st.markdown('<div class="warn-box">⚠️ Please enter your Groq API key in the sidebar.</div>',
                    unsafe_allow_html=True)
    else:
        quick_qs = {
            "😴 Why am I so tired?":  "I have Hashimoto's and I'm exhausted even after 8 hours sleep. Why?",
            "🧠 Brain fog tips":       "What helps with Hashimoto's brain fog?",
            "🍽️ What to eat?":        "What's the best diet for Hashimoto's? Should I go gluten free?",
            "💊 Medication timing":    "When and how should I take Levothyroxine for best absorption?",
            "🧪 Understanding TSH":    "My TSH is 3.5, what does that mean for Hashimoto's?",
            "💪 Exercise tips":        "What exercise is safe and helpful with Hashimoto's?",
            "😰 Stress & thyroid":     "How does stress affect my Hashimoto's?",
            "👩‍⚕️ Doctor questions":  "What questions should I ask my endocrinologist?"
        }
        cols = st.columns(4)
        for i, (label, question) in enumerate(quick_qs.items()):
            with cols[i % 4]:
                if st.button(label, use_container_width=True):
                    st.session_state.hashi_quick_q = question

        st.divider()

        if "hashi_messages" not in st.session_state:
            st.session_state.hashi_messages = []
            with st.spinner("Starting your coach..."):
                greeting = chat([], "Introduce yourself warmly as a Hashimoto's coach. Keep it brief and friendly.")
            st.session_state.hashi_messages.append({"role": "assistant", "content": greeting})

        for msg in st.session_state.hashi_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if "hashi_quick_q" in st.session_state:
            q = st.session_state.pop("hashi_quick_q")
            st.session_state.hashi_messages.append({"role": "user", "content": q})
            with st.spinner("Thinking..."):
                r = chat(st.session_state.hashi_messages[:-1], q)
            st.session_state.hashi_messages.append({"role": "assistant", "content": r})
            st.rerun()

        if user_input := st.chat_input("Ask your Hashimoto coach..."):
            st.session_state.hashi_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    r = chat(st.session_state.hashi_messages[:-1], user_input)
                st.write(r)
            st.session_state.hashi_messages.append({"role": "assistant", "content": r})

# ═════════════════════════════════════════════════════
# TAB 5 — Supplements
# ═════════════════════════════════════════════════════
with tab5:
    st.header("💊 Supplement Tracker")
    supps = load_supplements(client)

    with st.expander("📚 Common Hashimoto Supplements & Why"):
        ref_supps = pd.DataFrame([
            ["Selenium",      "200 mcg/day",  "Reduces TPO antibodies, supports T4 to T3 conversion"],
            ["Vitamin D3",    "2000-5000 IU", "Often deficient; immune modulation"],
            ["Magnesium",     "300-400 mg",   "Reduces fatigue, improves sleep, supports thyroid"],
            ["Zinc",          "15-30 mg",     "Supports T4 to T3 conversion"],
            ["Vitamin B12",   "1000 mcg",     "Often low in Hashi; energy and nerve function"],
            ["Iron/Ferritin", "Per lab",      "Low ferritin worsens fatigue and hair loss"],
            ["Omega-3",       "1-2g EPA+DHA", "Anti-inflammatory, reduces antibodies"],
            ["Ashwagandha",   "300-600 mg",   "Adaptogen for stress and cortisol (use cautiously)"],
            ["Inositol",      "600 mg",       "May help reduce TSH and antibodies"],
        ], columns=["Supplement", "Typical Dose", "Why It Helps"])
        st.dataframe(ref_supps, width="stretch")

    st.divider()
    st.subheader("➕ Add Supplement")
    col1, col2, col3 = st.columns(3)
    with col1:
        supp_name = st.text_input("Supplement name", placeholder="e.g. Selenium")
    with col2:
        supp_dose = st.text_input("Dose", placeholder="e.g. 200mcg")
    with col3:
        supp_time = st.selectbox("When to take",
                        ["Morning", "Midday", "Evening", "With food", "Bedtime"])
    supp_notes = st.text_input("Notes (brand, form, etc.)")

    if st.button("➕ Add Supplement", use_container_width=True):
        if supp_name:
            supp_id = supp_name.lower().replace(" ", "_")
            supps[supp_id] = {
                "name": supp_name, "dose": supp_dose,
                "time": supp_time, "notes": supp_notes,
                "active": True, "added": str(date.today())
            }
            with st.spinner("Saving..."):
                save_supplement(supps, client)
                st.cache_data.clear()
            st.success(f"✅ {supp_name} added!")
            st.rerun()

    st.divider()
    st.subheader("✅ Today's Check-off")
    today_key = f"supp_taken_{date.today()}"

    if not supps:
        st.info("No supplements added yet.")
    else:
        taken_today  = st.session_state.get(today_key, {})
        active_supps = {k: v for k, v in supps.items() if str(v.get("active", True)) == "True"}

        for supp_id, info in active_supps.items():
            c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
            with c1:
                taken = st.checkbox(f"{info['name']} — {info.get('dose','')}",
                                    value=taken_today.get(supp_id, False),
                                    key=f"check_{supp_id}")
                taken_today[supp_id] = taken
            with c2:
                st.caption(f"🕐 {info.get('time','')}")
            with c3:
                st.caption(info.get("notes", ""))
            with c4:
                if st.button("🗑️", key=f"del_{supp_id}"):
                    supps[supp_id]["active"] = False
                    save_supplement(supps, client)
                    st.rerun()

        st.session_state[today_key] = taken_today
        total   = len(active_supps)
        checked = sum(1 for v in taken_today.values() if v)
        pct     = int(checked / total * 100) if total else 0
        st.progress(pct / 100, text=f"{checked}/{total} taken today ({pct}%)")

    st.divider()
    if GROQ_API_KEY:
        st.subheader("🤖 Ask About Supplements")
        if supp_q := st.chat_input("Ask about supplements...", key="supp_chat"):
            with st.spinner("Researching..."):
                r = ask(f"Hashimoto's health question about supplements: {supp_q}. Be specific, evidence-based, remind user to check with doctor.")
            st.info(r)

# ═════════════════════════════════════════════════════
# TAB 6 — Flare Alerts
# ═════════════════════════════════════════════════════
with tab6:
    st.header("🚨 Flare Alert Centre")

    with st.expander("ℹ️ How flare detection works"):
        thresh_df = pd.DataFrame([
            ["Energy",       "4 or below",  "Very low energy may signal flare or under-medication"],
            ["Brain Fog",    "7 or above",  "Severe cognitive symptoms spike during flares"],
            ["Joint Pain",   "7 or above",  "Inflammation marker common in Hashimoto's flares"],
            ["Mood",         "4 or below",  "Low mood can indicate thyroid hormone fluctuation"],
            ["Stress Level", "8 or above",  "High stress triggers immune response and flares"],
        ], columns=["Symptom", "Threshold", "Why It Matters"])
        st.dataframe(thresh_df, width="stretch")
        st.caption("Personal tracking tool only. Always consult your doctor.")

    st.divider()
    st.subheader("⚙️ Customise Thresholds")
    col1, col2, col3 = st.columns(3)
    with col1:
        t_energy = st.slider("Energy alert if below",     1, 5,  4)
        t_mood   = st.slider("Mood alert if below",       1, 5,  4)
    with col2:
        t_fog    = st.slider("Brain fog alert if above",  5, 10, 7)
        t_stress = st.slider("Stress alert if above",     5, 10, 8)
    with col3:
        t_joint  = st.slider("Joint pain alert if above", 5, 10, 7)

    FLARE_THRESHOLDS.update({
        "energy":       ("low",  t_energy),
        "mood":         ("low",  t_mood),
        "brain_fog":    ("high", t_fog),
        "stress_level": ("high", t_stress),
        "joint_pain":   ("high", t_joint),
    })

    st.divider()
    if st.button("🚨 Run Flare Check on Today", use_container_width=True):
        all_logs  = load_daily_logs(client)
        today_log = all_logs.get(str(date.today()))
        if not today_log:
            st.warning("No log for today yet. Save your daily log first!")
        else:
            alerts = detect_flare(today_log)
            if alerts:
                st.error("🚨 Flare indicators detected!")
                for a in alerts:
                    st.markdown(f'<div class="flare-box">⚠️ {a}</div>', unsafe_allow_html=True)
                if GROQ_API_KEY:
                    with st.spinner("Getting tips..."):
                        advice = ask(f"""Hashimoto patient has these flare symptoms today:
                        {', '.join(alerts)}
                        Give 5 immediate practical things they can do to manage this flare.
                        Be empathetic. Remind them to contact doctor if severe.""")
                    st.divider()
                    st.subheader("💡 Flare Management Tips")
                    st.write(advice)
            else:
                st.markdown('<div class="ok-box">✅ No flare indicators today!</div>',
                            unsafe_allow_html=True)

    st.divider()
    st.subheader("📅 Flare History")
    flares = load_flares(client)

    if not flares:
        st.info("No flares recorded yet.")
    else:
        flare_list = sorted(flares.items(), reverse=True)
        st.metric("Total flares recorded", len(flare_list))

        for ts, flare in flare_list[:10]:
            with st.expander(f"🚨 {flare['date']} — {len(flare['alerts'])} alert(s)"):
                for a in flare["alerts"]:
                    st.markdown(f'<div class="flare-box">⚠️ {a}</div>', unsafe_allow_html=True)

        if len(flares) >= 2:
            st.subheader("📊 Flare Frequency")
            flare_dates = [v["date"] for v in flares.values()]
            flare_count = pd.Series(flare_dates).value_counts().reset_index()
            flare_count.columns = ["date", "flares"]
            flare_count = flare_count.sort_values("date")
            fig = px.bar(flare_count, x="date", y="flares", title="Flares Per Day",
                         color="flares", color_continuous_scale=["#4fa3a0","#e07b7b"])
            st.plotly_chart(fig, use_container_width=True)

        if GROQ_API_KEY:
            st.divider()
            if st.button("🧠 Analyse My Flare Patterns"):
                all_logs = load_daily_logs(client)
                with st.spinner("Analysing..."):
                    result = ask(f"""Analyse this Hashimoto patient's flare history.
                    Flares recorded: {len(flares)}
                    Flare dates: {[v['date'] for v in flares.values()]}
                    Recent logs: {json.dumps(dict(list(all_logs.items())[-7:]), indent=2)}
                    Identify: triggers, patterns, what helps, recommendations, doctor questions.
                    Be specific and empathetic.""")
                st.write(result)

# ═════════════════════════════════════════════════════
# TAB 7 — Doctor Report
# ═════════════════════════════════════════════════════
with tab7:
    st.header("📄 Generate Doctor Report")

    if not GROQ_API_KEY:
        st.markdown('<div class="warn-box">⚠️ Enter your Groq API key in the sidebar.</div>',
                    unsafe_allow_html=True)
    else:
        days_back = st.slider("Include last X days", 7, 90, 30)

        if st.button("📄 Generate Report", use_container_width=True):
            all_logs = load_daily_logs(client)
            labs     = load_labs(client)
            flares   = load_flares(client)
            supps    = load_supplements(client)

            recent = list(all_logs.items())[-days_back:]
            if not recent:
                st.warning("No logs found. Start logging daily first!")
            else:
                df = pd.DataFrame([v for _, v in recent])
                for col in ["energy","mood","brain_fog","sleep_hrs",
                            "weight","stress_level","joint_pain","hydration"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                active_supps  = [v["name"] + " " + str(v.get("dose",""))
                                  for v in supps.values()
                                  if str(v.get("active","True")) == "True"]
                avg_energy    = round(df["energy"].mean(), 1)       if "energy"       in df.columns else "N/A"
                avg_mood      = round(df["mood"].mean(), 1)         if "mood"         in df.columns else "N/A"
                avg_fog       = round(df["brain_fog"].mean(), 1)    if "brain_fog"    in df.columns else "N/A"
                avg_sleep     = round(df["sleep_hrs"].mean(), 1)    if "sleep_hrs"    in df.columns else "N/A"
                avg_stress    = round(df["stress_level"].mean(), 1) if "stress_level" in df.columns else "N/A"
                avg_hydration = round(df["hydration"].mean(), 1)    if "hydration"    in df.columns else "N/A"
                latest_labs   = labs[-1] if labs else "Not recorded"

                with st.spinner("Generating report..."):
                    report = ask(f"""Create a professional medical summary for a Hashimoto's patient
                    to share with their endocrinologist.

                    Period: {recent[0][0]} to {recent[-1][0]}
                    Days logged: {len(recent)}
                    Average energy: {avg_energy}/10
                    Average mood: {avg_mood}/10
                    Average brain fog: {avg_fog}/10
                    Average sleep: {avg_sleep} hours/night
                    Average stress: {avg_stress}/10
                    Average hydration: {avg_hydration} glasses/day
                    Total flares this period: {len(flares)}
                    Current supplements: {', '.join(active_supps) if active_supps else 'None recorded'}
                    Latest labs: {latest_labs}

                    Format with these sections:
                    1. Patient Overview
                    2. Symptom Summary & Key Patterns
                    3. Sleep & Energy Analysis
                    4. Flare History & Triggers
                    5. Diet & Lifestyle Compliance
                    6. Current Supplement Protocol
                    7. Latest Lab Values & Interpretation
                    8. Recommended Discussion Points for Doctor

                    Be professional, specific and empathetic.""")

                st.divider()
                st.subheader("Your Doctor Report")
                st.write(report)
                st.download_button(
                    "⬇️ Download Report as TXT",
                    data=report,
                    file_name=f"hashimoto_report_{date.today()}.txt",
                    use_container_width=True
                )
