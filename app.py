import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

OUT_DIR = Path("outputs")
CLASSIFIED_CSV = OUT_DIR / "classified_emails.csv"
STATE_CSV = OUT_DIR / "state.csv"

st.set_page_config(page_title="AI-Powered Communication Assistant", layout="wide")

st.title("📬 AI-Powered Communication Assistant")
st.caption("Filter • Prioritize • Extract • Draft Replies")

# Load
if not CLASSIFIED_CSV.exists():
    st.error("Run `python main.py` first to generate outputs/classified_emails.csv")
    st.stop()

df = pd.read_csv(CLASSIFIED_CSV, parse_dates=["received_at"])
state = pd.read_csv(STATE_CSV) if STATE_CSV.exists() else pd.DataFrame(columns=["sender","subject","status"])

# Merge status
df = df.merge(state, on=["sender","subject"], how="left")
df["status"] = df["status"].fillna("Pending")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    sentiments = st.multiselect("Sentiment", ["Positive","Neutral","Negative"], default=["Positive","Neutral","Negative"])
    priorities = st.multiselect("Priority", ["Urgent","Not urgent"], default=["Urgent","Not urgent"])
    categories = st.multiselect("Category", ["Billing Issue","Account Issue","Technical/Integration","General Query"],
                                default=["Billing Issue","Account Issue","Technical/Integration","General Query"])
    status_filter = st.multiselect("Status", ["Pending","Resolved"], default=["Pending","Resolved"])
    last_hrs = st.slider("Received in last (hours)", 0, 72, 72)

    st.markdown("---")
    if st.button("Save Status Changes"):
        # Persist status back
        df[["sender","subject","status"]].to_csv(STATE_CSV, index=False)
        st.success("Saved ✅")

# Apply filters
cutoff = datetime.now() - timedelta(hours=last_hrs)
mask = (
    df["sentiment"].isin(sentiments) &
    df["priority"].isin(priorities) &
    df["category"].isin(categories) &
    df["status"].isin(status_filter) 
)

view = df[mask].copy()

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total (filtered)", len(view))
col2.metric("Urgent", (view["priority"] == "Urgent").sum())
col3.metric("Resolved", (view["status"] == "Resolved").sum())
col4.metric("Pending", (view["status"] == "Pending").sum())

# Simple analytics
st.subheader("Analytics")
c1, c2, c3 = st.columns(3)
with c1:
    st.bar_chart(view["category"].value_counts())
with c2:
    st.bar_chart(view["sentiment"].value_counts())
with c3:
    st.bar_chart(view["priority"].value_counts())

st.subheader("Filtered Support Emails")
# Editable status
for idx, row in view.iterrows():
    with st.expander(f"{row['priority']} • {row['category']} • {row['subject']}  —  {row['sender']}"):
        st.write(f"**Received:** {row['received_at']}")
        st.write("**Body:**")
        st.write(row["body"])
        st.write("**Extracted Info:**")
        st.json({
            "emails_found": eval(row["emails_found"]) if isinstance(row["emails_found"], str) else row["emails_found"],
            "phones_found": eval(row["phones_found"]) if isinstance(row["phones_found"], str) else row["phones_found"],
            "possible_requirements": eval(row["requirements"]) if isinstance(row["requirements"], str) else row["requirements"],
        })
        st.write("**Sentiment:**", row["sentiment"])
        st.write("**Priority:**", row["priority"])
        st.write("**Category:**", row["category"])

        # Draft reply (editable before sending)
        reply_key = f"reply_{idx}"
        draft = st.text_area("AI-Generated Reply (edit before sending):", value=row["draft_reply"], key=reply_key, height=220)

        # Status toggle
        new_status = st.radio("Mark status:", ["Pending","Resolved"], index=0 if row["status"]=="Pending" else 1, key=f"status_{idx}")
        df.at[idx, "status"] = new_status

        # (Optional) Send button (stub)
        if st.button("Send Reply (stub)", key=f"send_{idx}"):
            st.info("This is a demo. Wire this to Gmail/SMTP/Outlook to actually send.")
            st.success("Queued for sending (demo).")

# Persist status in-memory until Save clicked
st.caption("Click **Save Status Changes** in the sidebar to persist updates.")
