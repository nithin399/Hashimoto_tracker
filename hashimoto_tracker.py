import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import os
import pandas as pd
import plotly.express as px
from datetime import datetime, date

# ── Model (Groq Cloud) ────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def get_llm():
    return ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        api_key=GROQ_API_KEY,
        temperature=0.7
    )

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

You know about TSH, Free T3, Free T4, Total T3, Total T4, TPO antibodies, TgAb, RA Factor,
Levothyroxine, NDT, T3/T4 combo therapy,
gluten-thyroid connection, AIP diet, anti-inflammatory foods, histamine intolerance,
adrenal fatigue, cortisol, stress impact,
sleep, exercise, supplements like Selenium, Vitamin D, Vitamin B12, Magnesium,
brain fog, fatigue, weight gain, hair loss, cold intolerance,
menstrual irregularities, constipation, soy and food sensitivities."""

def chat(messages, user_input):
    llm = get_llm()
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in messages:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=user_input))
    response = llm.invoke(msgs)
    return response.content

def ask(prompt_text):
    llm = get_llm()
    msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt_text)]
    return llm.invoke(msgs).content

# ── Storage ───────────────────────────────────────────
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

def save_daily_log(data):
    logs = load_json(LOGS_FILE)
    logs[str(date.today())] = data
    save_json(LOGS_FILE, logs)

def save_lab_result(data):
    labs = load_json(LABS_FILE)
    labs[str(datetime.now())] = data
    save_json(LABS_FILE, labs)

# ── Flare detection ───────────────────────────────────
FLARE_THRESHOLDS = {
    "energy":       ("low",  4),
    "brain_fog":    ("high", 7),
    "joint_pain":   ("high", 7),
    "mood":         ("low",  4),
    "stress_level": ("high", 8),
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

def save_flare(log, alerts):
    flares = load_json(FLARES_FILE)
    flares[str(datetime.now())] = {
        "date": str(date.today()),
        "alerts": alerts,
        "snapshot": {k: log.get(k) for k in FLARE_THRESHOLDS}
    }
    save_json(FLARES_FILE, flares)

# ── Page config ───────────────────────────────────────
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
  .api-warning {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    padding: 0.8rem 1.2rem;
    border-radius: 6px;
    margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)

st.title("🦋 Hashimoto Tracker & Coach")
st.caption("Private · Secure · Your personal health companion")

# ── API Key check ─────────────────────────────────────
if not GROQ_API_KEY:
    st.markdown("""
    <div class="api-warning">
    ⚠️ <b>Groq API key not found.</b> Add your key in the sidebar to enable AI features.
    Get a free key at <a href="https://console.groq.com" target="_blank">console.groq.com</a>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar API key input ─────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    if not GROQ_API_KEY:
        user_key = st.text_input("Groq API Key", type="password",
                                  placeholder="gsk_...")
        if user_key:
            GROQ_API_KEY = user_key
            os.environ["GROQ_API_KEY"] = user_key
            st.success("API key set!")
    else:
        st.success("✅ AI Connected")
    st.caption("Get free key at console.groq.com")

# ── Tabs ──────────────────────────────────────────────
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
    logs     = load_json(LOGS_FILE)
    existing = logs.get(today, {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("😴 Energy & Mood")
        energy        = st.slider("Energy level", 1, 10, int(existing.get("energy", 5)))
        mood          = st.slider("Mood", 1, 10, int(existing.get("mood", 5)))
        brain_fog     = st.slider("Brain fog (1=none, 10=severe)", 1, 10, int(existing.get("brain_fog", 1)))
        sleep_hrs     = st.number_input("Hours slept", 0.0, 12.0, float(existing.get("sleep_hrs", 7.0)), 0.5)
        sleep_quality = st.slider("Sleep quality", 1, 10, int(existing.get("sleep_quality", 5)))
        bed_time      = st.time_input("🌙 Off to Bed time",
                            datetime.strptime(existing.get("bed_time", "22:30"), "%H:%M").time())
        wake_time     = st.time_input("☀️ Up from Bed time",
                            datetime.strptime(existing.get("wake_time", "07:00"), "%H:%M").time())

    with col2:
        st.subheader("🌡️ Physical Symptoms")
        weight = st.number_input("Weight (kg)", 30.0, 200.0, float(existing.get("weight", 65.0)), 0.1)
        temp_opts        = ["Normal", "Cold", "Hot", "Both"]
        temp_sensitivity = st.selectbox("Temperature sensitivity", temp_opts,
                               index=temp_opts.index(existing.get("temp_sensitivity", "Normal")))
        hair_opts = ["None", "Mild", "Moderate", "Severe"]
        hair_loss = st.selectbox("Hair loss today", hair_opts,
                        index=hair_opts.index(existing.get("hair_loss", "None")))
        joint_pain    = st.slider("Joint/muscle pain (1=none)", 1, 10, int(existing.get("joint_pain", 1)))
        bloating      = st.slider("Bloating/digestion (1=none)", 1, 10, int(existing.get("bloating", 1)))
        constipation_opts = ["None", "Mild", "Moderate", "Severe"]
        constipation  = st.selectbox("Constipation today", constipation_opts,
                            index=constipation_opts.index(existing.get("constipation", "None")))
        menstrual_notes = st.text_area("🌸 Menstrual Cycle Notes",
                            existing.get("menstrual_notes", ""),
                            placeholder="e.g. Day 3, heavy flow, cramps, spotting...")

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("💊 Medication")
        meds      = load_json(MEDS_FILE)
        med_name  = st.text_input("Medication name", meds.get("name", "Levothyroxine"))
        med_dose  = st.text_input("Dose", meds.get("dose", "50mcg"))
        med_taken = st.checkbox("Taken today (fasting)?", bool(existing.get("med_taken", False)))
        med_time  = st.time_input("Time taken",
                       datetime.strptime(existing.get("med_time", "07:00"), "%H:%M").time())
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
        notes            = st.text_area("Notes / symptoms", existing.get("notes", ""))

        st.subheader("🏃 Exercise Today")
        ex_walking = st.checkbox("🚶 Walking", bool(existing.get("ex_walking", False)))
        ex_weights = st.checkbox("🏋️ Weights", bool(existing.get("ex_weights", False)))
        ex_yoga    = st.checkbox("🧘 Yoga",    bool(existing.get("ex_yoga", False)))
        ex_other   = st.text_input("Other exercise", existing.get("ex_other", ""))

    if st.button("✅ Save Today's Log", use_container_width=True):
        new_log = dict(
            energy=energy, mood=mood, brain_fog=brain_fog,
            sleep_hrs=sleep_hrs, sleep_quality=sleep_quality,
            bed_time=str(bed_time), wake_time=str(wake_time),
            weight=weight, temp_sensitivity=temp_sensitivity,
            hair_loss=hair_loss, joint_pain=joint_pain,
            bloating=bloating, constipation=constipation,
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
        save_daily_log(new_log)
        alerts = detect_flare(new_log)
        if alerts:
            save_flare(new_log, alerts)
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
        tsh      = st.number_input("TSH (mIU/L)",       0.0, 20.0,   2.5, 0.01)
        free_t4  = st.number_input("Free T4 (pmol/L)",  0.0, 40.0,  15.0, 0.1)
        free_t3  = st.number_input("Free T3 (pmol/L)",  0.0, 15.0,   5.0, 0.1)
        total_t3 = st.number_input("Total T3 (nmol/L)", 0.0, 10.0,   1.8, 0.01)
        total_t4 = st.number_input("Total T4 (nmol/L)", 0.0, 250.0, 100.0, 0.1)
    with col2:
        st.subheader("Antibodies & Other")
        tpo_ab   = st.number_input("TPO Antibodies (IU/mL)", 0.0, 5000.0, 100.0, 1.0)
        tg_ab    = st.number_input("TgAb (IU/mL)",           0.0, 5000.0,  50.0, 1.0)
        ra_factor= st.number_input("RA Factor (IU/mL)",      0.0,  500.0,  10.0, 0.1)
        vit_d    = st.number_input("Vitamin D (nmol/L)",     0.0,  200.0,  50.0, 1.0)
        vit_b12  = st.number_input("Vitamin B12 (pmol/L)",   0.0, 2000.0, 300.0, 1.0)
        ferritin = st.number_input("Ferritin (ug/L)",        0.0,  500.0,  70.0, 1.0)

    lab_notes = st.text_area("Lab notes")
    lab_date  = st.date_input("Lab date", date.today())

    if st.button("💾 Save Lab Results", use_container_width=True):
        save_lab_result(dict(
            date=str(lab_date), tsh=tsh, free_t4=free_t4,
            free_t3=free_t3, total_t3=total_t3, total_t4=total_t4,
            tpo_ab=tpo_ab, tg_ab=tg_ab, ra_factor=ra_factor,
            vit_d=vit_d, vit_b12=vit_b12, ferritin=ferritin,
            notes=lab_notes
        ))
        st.success("Lab results saved!")

    st.divider()
    st.subheader("📊 Reference Ranges")
    ref = pd.DataFrame({
        "Marker": ["TSH", "Free T4", "Free T3", "Total T3", "Total T4",
                   "TPO Antibodies", "RA Factor", "Vitamin D", "Vitamin B12", "Ferritin"],
        "Optimal (Hashimoto's)": [
            "1-2 mIU/L", "15-23 pmol/L", "5-7 pmol/L",
            "1.5-2.5 nmol/L", "90-120 nmol/L",
            "<35 IU/mL", "<14 IU/mL", ">100 nmol/L", ">300 pmol/L", "70-150 ug/L"],
        "Standard Lab Range": [
            "0.4-4.0", "9-19 pmol/L", "3.5-6.5 pmol/L",
            "1.2-2.7 nmol/L", "60-150 nmol/L",
            "<34 IU/mL", "<14 IU/mL", "50-250 nmol/L", "145-637 pmol/L", "12-300 ug/L"]
    })
    st.dataframe(ref, width="stretch")

# ═════════════════════════════════════════════════════
# TAB 3 — Trends
# ═════════════════════════════════════════════════════
with tab3:
    st.header("📊 Your Trends")
    logs = load_json(LOGS_FILE)
    if len(logs) < 2:
        st.info("Log at least 2 days to see trends.")
    else:
        df = pd.DataFrame(logs).T.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"])
        num_cols = ["energy","mood","brain_fog","sleep_hrs","weight",
                    "stress_level","joint_pain","bloating","hydration"]
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
                fig5 = px.bar(df, x="date", y="hydration", title="Daily Hydration (glasses)",
                              color="hydration", color_continuous_scale="Blues")
                st.plotly_chart(fig5, use_container_width=True)
        with col2:
            if "weight" in df.columns:
                fig3 = px.line(df, x="date", y="weight", title="Weight (kg)", line_shape="spline")
                fig3.update_traces(line_color="#9b59b6")
                st.plotly_chart(fig3, use_container_width=True)
            if all(c in df.columns for c in ["stress_level","energy"]):
                fig4 = px.scatter(df, x="stress_level", y="energy",
                                  title="Stress vs Energy", trendline="ols",
                                  color="mood", color_continuous_scale="RdYlGn")
                st.plotly_chart(fig4, use_container_width=True)

# ═════════════════════════════════════════════════════
# TAB 4 — AI Coach
# ═════════════════════════════════════════════════════
with tab4:
    st.header("💬 Hashimoto Coach")
    st.caption("Ask anything about your condition — powered by Groq AI")

    if not GROQ_API_KEY:
        st.warning("Please enter your Groq API key in the sidebar to use the AI Coach.")
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
                greeting = chat([], "Introduce yourself warmly as a Hashimoto's coach. Acknowledge that Hashimoto's is challenging and you are here to help. Keep it brief.")
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
    st.caption("Track supplements that support thyroid health")

    supps = load_json(SUPPLEMENTS_FILE)

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
    st.subheader("➕ Add / Update Supplement")
    col1, col2, col3 = st.columns(3)
    with col1:
        supp_name = st.text_input("Supplement name", placeholder="e.g. Selenium")
    with col2:
        supp_dose = st.text_input("Dose", placeholder="e.g. 200mcg")
    with col3:
        supp_time = st.selectbox("When to take", ["Morning", "Midday", "Evening", "With food", "Bedtime"])
    supp_notes = st.text_input("Notes (brand, form, etc.)")

    if st.button("➕ Add Supplement", use_container_width=True):
        if supp_name:
            supp_id = supp_name.lower().replace(" ", "_")
            supps[supp_id] = {
                "name": supp_name, "dose": supp_dose,
                "time": supp_time, "notes": supp_notes,
                "active": True, "added": str(date.today())
            }
            save_json(SUPPLEMENTS_FILE, supps)
            st.success(f"{supp_name} added!")
            st.rerun()

    st.divider()
    st.subheader("✅ Today's Supplement Check-off")
    today_key = f"supp_taken_{date.today()}"

    if not supps:
        st.info("No supplements added yet. Add your first one above!")
    else:
        taken_today  = st.session_state.get(today_key, {})
        active_supps = {k: v for k, v in supps.items() if v.get("active", True)}

        for supp_id, info in active_supps.items():
            c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
            with c1:
                taken = st.checkbox(f"{info['name']} — {info['dose']}",
                                    value=taken_today.get(supp_id, False),
                                    key=f"check_{supp_id}")
                taken_today[supp_id] = taken
            with c2:
                st.caption(f"🕐 {info['time']}")
            with c3:
                st.caption(info.get("notes", ""))
            with c4:
                if st.button("🗑️", key=f"del_{supp_id}"):
                    supps[supp_id]["active"] = False
                    save_json(SUPPLEMENTS_FILE, supps)
                    st.rerun()

        st.session_state[today_key] = taken_today
        total   = len(active_supps)
        checked = sum(1 for v in taken_today.values() if v)
        pct     = int(checked / total * 100) if total else 0
        st.progress(pct / 100, text=f"{checked}/{total} supplements taken today ({pct}%)")

    st.divider()
    st.subheader("🤖 Ask About Supplements")
    if not GROQ_API_KEY:
        st.warning("Enter your Groq API key in the sidebar to use this feature.")
    elif supp_q := st.chat_input("Ask about supplements...", key="supp_chat"):
        with st.spinner("Researching..."):
            r = ask(f"As a Hashimoto's health coach, answer this supplement question: {supp_q}. Be specific, evidence-based, and remind user to check with their doctor.")
        st.info(r)

# ═════════════════════════════════════════════════════
# TAB 6 — Flare Alerts
# ═════════════════════════════════════════════════════
with tab6:
    st.header("🚨 Flare Alert Centre")
    st.caption("Automatically detected when your symptoms cross warning thresholds")

    with st.expander("ℹ️ How flare detection works"):
        thresh_df = pd.DataFrame([
            ["Energy",       "4 or below",  "Very low energy may signal flare or under-medication"],
            ["Brain Fog",    "7 or above",  "Severe cognitive symptoms often spike during flares"],
            ["Joint Pain",   "7 or above",  "Inflammation marker common in Hashimoto's flares"],
            ["Mood",         "4 or below",  "Low mood can indicate thyroid hormone fluctuation"],
            ["Stress Level", "8 or above",  "High stress triggers immune response and flares"],
        ], columns=["Symptom", "Threshold", "Why It Matters"])
        st.dataframe(thresh_df, width="stretch")
        st.caption("This is a personal tracking tool only. Always consult your doctor.")

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
    st.subheader("🔍 Check Today's Log for Flare")
    if st.button("🚨 Run Flare Check on Today", use_container_width=True):
        logs      = load_json(LOGS_FILE)
        today_log = logs.get(str(date.today()))
        if not today_log:
            st.warning("No log for today yet. Save your daily log first!")
        else:
            alerts = detect_flare(today_log)
            if alerts:
                st.error("Flare indicators detected!")
                for a in alerts:
                    st.markdown(f'<div class="flare-box">⚠️ {a}</div>', unsafe_allow_html=True)
                if GROQ_API_KEY:
                    with st.spinner("Getting flare management tips..."):
                        advice = ask(f"""A Hashimoto patient is showing these flare symptoms today:
                        {', '.join(alerts)}
                        Give 5 immediate practical things they can do today to manage this flare.
                        Be empathetic and specific. Remind them to contact their doctor if severe.""")
                    st.divider()
                    st.subheader("💡 Flare Management Tips")
                    st.write(advice)
            else:
                st.markdown('<div class="ok-box">✅ No flare indicators today — you are doing well!</div>',
                            unsafe_allow_html=True)

    st.divider()
    st.subheader("📅 Flare History")
    flares = load_json(FLARES_FILE)

    if not flares:
        st.info("No flares recorded yet. Flares are auto-detected when you save your daily log.")
    else:
        flare_list = sorted(flares.items(), reverse=True)
        st.metric("Total flares recorded", len(flare_list))

        for ts, flare in flare_list[:10]:
            with st.expander(f"🚨 {flare['date']} — {len(flare['alerts'])} alert(s)"):
                for a in flare["alerts"]:
                    st.markdown(f'<div class="flare-box">⚠️ {a}</div>', unsafe_allow_html=True)
                snap = flare.get("snapshot", {})
                if snap:
                    st.dataframe(pd.DataFrame([snap]), width="stretch")

        if len(flares) >= 2:
            st.subheader("📊 Flare Frequency Chart")
            flare_dates = [v["date"] for v in flares.values()]
            flare_count = pd.Series(flare_dates).value_counts().reset_index()
            flare_count.columns = ["date", "flares"]
            flare_count = flare_count.sort_values("date")
            fig = px.bar(flare_count, x="date", y="flares", title="Flares Per Day",
                         color="flares", color_continuous_scale=["#4fa3a0", "#e07b7b"])
            st.plotly_chart(fig, use_container_width=True)

        if GROQ_API_KEY:
            st.divider()
            st.subheader("🧠 AI Flare Pattern Analysis")
            if st.button("🔍 Analyse My Flare Patterns"):
                logs   = load_json(LOGS_FILE)
                flares = load_json(FLARES_FILE)
                with st.spinner("Analysing patterns..."):
                    result = ask(f"""Analyse this Hashimoto patient flare history and daily logs.
                    Number of flares recorded: {len(flares)}
                    Flare dates: {[v['date'] for v in flares.values()]}
                    Recent logs last 7 entries: {json.dumps(dict(list(logs.items())[-7:]), indent=2)}
                    Identify:
                    1. Patterns — what conditions precede flares?
                    2. Possible triggers based on the data
                    3. What seems to help on better days
                    4. Recommendations to reduce flare frequency
                    5. Questions to discuss with their doctor
                    Be specific and empathetic.""")
                st.write(result)

# ═════════════════════════════════════════════════════
# TAB 7 — Doctor Report
# ═════════════════════════════════════════════════════
with tab7:
    st.header("📄 Generate Doctor Report")
    days_back = st.slider("Include last X days", 7, 90, 30)

    if not GROQ_API_KEY:
        st.warning("Enter your Groq API key in the sidebar to generate a report.")
    elif st.button("📄 Generate Report", use_container_width=True):
        logs   = load_json(LOGS_FILE)
        labs   = load_json(LABS_FILE)
        flares = load_json(FLARES_FILE)
        supps  = load_json(SUPPLEMENTS_FILE)

        recent = list(logs.items())[-days_back:]
        if not recent:
            st.warning("No logs found. Start logging daily first!")
        else:
            df = pd.DataFrame([v for _, v in recent])
            for col in ["energy","mood","brain_fog","sleep_hrs","weight","stress_level","joint_pain","hydration"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            active_supps = [v["name"] + " " + v.get("dose", "") for v in supps.values() if v.get("active")]
            avg_energy = round(df["energy"].mean(), 1)       if "energy"       in df.columns else "N/A"
            avg_mood   = round(df["mood"].mean(), 1)         if "mood"         in df.columns else "N/A"
            avg_fog    = round(df["brain_fog"].mean(), 1)    if "brain_fog"    in df.columns else "N/A"
            avg_sleep  = round(df["sleep_hrs"].mean(), 1)    if "sleep_hrs"    in df.columns else "N/A"
            avg_stress = round(df["stress_level"].mean(), 1) if "stress_level" in df.columns else "N/A"
            avg_hydration = round(df["hydration"].mean(), 1) if "hydration"    in df.columns else "N/A"

            with st.spinner("Generating your doctor report..."):
                report = ask(f"""Create a clear medical summary report for a Hashimoto patient
                to share with their endocrinologist.
                Period: {recent[0][0]} to {recent[-1][0]}
                Average energy: {avg_energy}/10
                Average mood: {avg_mood}/10
                Average brain fog: {avg_fog}/10
                Average sleep: {avg_sleep} hours
                Average stress: {avg_stress}/10
                Average hydration: {avg_hydration} glasses/day
                Number of flares: {len(flares)}
                Current supplements: {', '.join(active_supps) if active_supps else 'None recorded'}
                Latest labs: {list(labs.values())[-1] if labs else 'Not recorded'}
                Format as a professional medical summary with:
                1. Overview of the period
                2. Key symptoms and patterns
                3. Flare frequency and triggers
                4. Supplement protocol
                5. Medication compliance
                6. Lifestyle factors including diet and hydration
                7. Suggested discussion points for the doctor""")

            st.divider()
            st.subheader("Your Doctor Report")
            st.write(report)
            st.download_button(
                "⬇️ Download Report",
                data=report,
                file_name=f"hashimoto_report_{date.today()}.txt",
                use_container_width=True
            )
