"""
app.py
------
Streamlit app for House Price Prediction + GenAI Agent (Milestone 2).
  - Property input form in a clean multi-column grid
  - Prediction + AI analysis results
  - Chat interface at the bottom
"""

import streamlit as st
from agent_graph import run_agent
from rag.advisor import answer_followup
from rag.retriever import retrieve_context

st.set_page_config(page_title="House Price AI Advisor", layout="wide")

# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_input" not in st.session_state:
    st.session_state.last_input = None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 House Price AI Advisor")
st.caption("Powered by XGBoost + LangGraph + Groq LLM")
st.markdown("---")

# ── Property Input Form ───────────────────────────────────────────────────────
st.subheader("📋 Property Details")

c1, c2, c3 = st.columns(3)

with c1:
    bedrooms   = st.number_input("🛏 Bedrooms",           min_value=0, max_value=20, value=3)
    sqft_living = st.number_input("📐 Living Area (sqft)", min_value=0, value=1800)
    floors     = st.number_input("🏢 Floors",             min_value=0, max_value=5,  value=1)
    waterfront = st.selectbox("🌊 Waterfront",            [0, 1])
    city       = st.text_input("🏙 City",                 value="Seattle")

with c2:
    bathrooms     = st.number_input("🚿 Bathrooms",          min_value=0, max_value=20, value=2)
    sqft_lot      = st.number_input("🌿 Lot Area (sqft)",    min_value=0, value=5000)
    sqft_above    = st.number_input("⬆ Sqft Above Ground",  min_value=0, value=1800)
    view          = st.selectbox("👁 View (0–4)",            [0, 1, 2, 3, 4])
    statezip      = st.text_input("📮 State + Zip",          value="WA 98178")

with c3:
    house_age         = st.number_input("🏚 House Age (years)",  min_value=0, value=20)
    sqft_basement     = st.number_input("🪜 Sqft Basement",      min_value=0, value=0)
    condition         = st.selectbox("⭐ Condition (1–5)",       [1, 2, 3, 4, 5])
    has_been_renovated = st.selectbox("🔨 Has Been Renovated",   [0, 1])

st.markdown("")
predict_btn = st.button("🔍 Predict & Analyse", use_container_width=True, type="primary")

if predict_btn:
    property_input = {
        "bedrooms":            int(bedrooms),
        "bathrooms":           float(bathrooms),
        "sqft_living":         int(sqft_living),
        "sqft_lot":            int(sqft_lot),
        "floors":              float(floors),
        "sqft_above":          int(sqft_above),
        "sqft_basement":       int(sqft_basement),
        "house_age":           int(house_age),
        "waterfront":          int(waterfront),
        "view":                int(view),
        "condition":           int(condition),
        "city":                city,
        "statezip":            statezip,
        "has_been_renovated":  int(has_been_renovated),
    }

    with st.spinner("Running agent pipeline…"):
        result = run_agent(property_input)

    if result["error"]:
        st.error(f"Error: {result['error']}")
    else:
        st.session_state.last_result = result
        st.session_state.last_input  = property_input
        st.session_state.chat_history = []

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.last_result and not st.session_state.last_result["error"]:
    result = st.session_state.last_result
    st.markdown("---")

    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric(label="💰 Predicted Price", value=f"${result['predicted_price']:,.0f}")
    with r2:
        st.success("Analysis complete — see AI insights below.")

    st.subheader("🤖 AI Analysis")
    st.markdown(result["analysis"])

# ── Chat Interface ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💬 Ask the AI Advisor")

if not st.session_state.last_result:
    st.info("Run a prediction first, then ask follow-up questions here.")
else:
    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask about this property, market trends, investment advice…")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                eda_ctx  = retrieve_context(user_question)
                response = answer_followup(
                    question=user_question,
                    property_details=st.session_state.last_input,
                    predicted_price=st.session_state.last_result["predicted_price"],
                    eda_context=eda_ctx,
                    chat_history=st.session_state.chat_history,
                )
            st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
