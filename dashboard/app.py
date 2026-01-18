import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
from datetime import datetime


st.set_page_config(
    page_title="AI Smart Email Triage System",
    page_icon="📧",
    layout="wide"
)


st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.stTabs [role="tab"] { font-size: 16px; padding: 10px; }
.stExpander { border-radius: 10px; border: 1px solid #ddd; }
.badge {
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 13px;
}
.high { background-color: #e74c3c; }
.medium { background-color: #f39c12; }
.low { background-color: #27ae60; }
</style>
""", unsafe_allow_html=True)


st.title("📧 AI Powered Smart Email Classifier for Enterprises")
st.caption("Automatic email categorization, priority assignment & routing")


with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/category_model.pkl", "rb") as f:
    category_model = pickle.load(f)

with open("models/urgency_model.pkl", "rb") as f:
    urgency_model = pickle.load(f)


source_df = pd.read_csv("data/processed/final_email_dataset.csv")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


STRONG_SPAM = ["win", "prize", "reward", "click here", "verify account"]
REQUEST_WORDS = ["request", "please", "help", "feature", "access", "reset"]
FEEDBACK_WORDS = ["thank you", "great", "excellent", "smooth"]
COMPLAINT_WORDS = ["refund", "failed", "not working", "error", "blocked", "deducted"]


if "live_emails" not in st.session_state:
    st.session_state.live_emails = pd.DataFrame(
        columns=["email_text", "category", "urgency", "agent", "timestamp"]
    )


if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 5:
    st.session_state.last_refresh = time.time()
    st.rerun()


def classify_email(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])

    ml_cat = category_model.predict(vector)[0]
    ml_urg = urgency_model.predict(vector)[0]

    if any(w in cleaned for w in STRONG_SPAM):
        return "spam", "low"
    elif any(w in cleaned for w in REQUEST_WORDS):
        return "request", "medium"
    elif any(w in cleaned for w in FEEDBACK_WORDS):
        return "feedback", "low"
    elif any(w in cleaned for w in COMPLAINT_WORDS):
        return "complaint", "high"
    else:
        return ml_cat, ml_urg


if st.sidebar.button("📥 Simulate Incoming Email"):
    row = source_df.sample(1)
    text = row["email_text"].values[0]
    category, urgency = classify_email(text)

    new_row = pd.DataFrame(
        [[
            text,
            category,
            urgency,
            "Unassigned",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]],
        columns=["email_text", "category", "urgency", "agent", "timestamp"]
    )

    st.session_state.live_emails = pd.concat(
        [new_row, st.session_state.live_emails],
        ignore_index=True
    )

    st.toast("📨 New email received and classified!", icon="📧")


df_live = st.session_state.live_emails

c_count = len(df_live[df_live["category"] == "complaint"])
r_count = len(df_live[df_live["category"] == "request"])
f_count = len(df_live[df_live["category"] == "feedback"])
s_count = len(df_live[df_live["category"] == "spam"])


col1, col2, col3 = st.columns(3)
col1.metric("📩 Total Emails", len(df_live))
col2.metric("🔥 High Priority", len(df_live[df_live["urgency"] == "high"]))
col3.metric("🚫 Spam Emails", s_count)


st.sidebar.markdown("### 👤 Agent Workload")
if not df_live.empty:
    st.sidebar.write(df_live["agent"].value_counts())
else:
    st.sidebar.write("No assignments yet")


st.markdown("---")
st.subheader("📂 Email Queues")

tab1, tab2, tab3, tab4 = st.tabs([
    f"🚨 Complaints ({c_count})",
    f"📝 Requests ({r_count})",
    f"💬 Feedback ({f_count})",
    f"🚫 Spam ({s_count})"
])

def urgency_badge(level):
    return f"<span class='badge {level}'>{level.upper()}</span>"

def render_table(category):
    subset = df_live[df_live["category"] == category]

    if subset.empty:
        st.info("📭 No emails in this queue.")
        return

    for i, row in subset.iterrows():
        preview = row["email_text"][:120] + "..."
        with st.expander(preview):
            st.write("📄 **Email:**", row["email_text"])
            st.markdown(
                f"⚠️ Urgency: {urgency_badge(row['urgency'])}",
                unsafe_allow_html=True
            )
            st.write("👤 **Agent:**", row["agent"])
            st.write("⏰ **Received:**", row["timestamp"])

            col1, col2, col3, col4 = st.columns(4)

            if col1.button("Agent A", key=f"a_{i}"):
                st.session_state.live_emails.at[i, "agent"] = "Agent A"

            if col2.button("Agent B", key=f"b_{i}"):
                st.session_state.live_emails.at[i, "agent"] = "Agent B"

            if col3.button("Agent C", key=f"c_{i}"):
                st.session_state.live_emails.at[i, "agent"] = "Agent C"

            if col4.button("✅ Resolved", key=f"r_{i}"):
                st.session_state.live_emails.drop(i, inplace=True)
                st.rerun()

with tab1:
    render_table("complaint")
with tab2:
    render_table("request")
with tab3:
    render_table("feedback")
with tab4:
    render_table("spam")


st.markdown("---")
st.subheader("📊 Live Analytics")

if not df_live.empty:
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        df_live["category"].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_title("Category Distribution")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        df_live["urgency"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", startangle=90, ax=ax2
        )
        ax2.set_ylabel("")
        ax2.set_title("Urgency Distribution")
        st.pyplot(fig2)
else:
    st.info("No emails received yet. Analytics will appear once emails arrive.")


st.markdown(
    "<hr><p style='text-align:center;color:gray;'>"
    "Enterprise AI Email Triage System • NLP • Machine Learning"
    "</p>",
    unsafe_allow_html=True
)
