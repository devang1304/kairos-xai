import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

# Adjust Full Page Width
st.set_page_config(layout="wide")

# Sidebar: Incident Filter
st.sidebar.header("Incident Filter")
alert_level = st.sidebar.radio(
    "Select Alert Level:",
    ["Critical", "High", "Medium"]
)

# Generate Timestamp and Incident ID
incident_id = "INC-2025-000124"
timestamp_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# Header Section
header_col1, header_col2, header_col3, header_col4 = st.columns(4)
with header_col1:
    st.markdown(f"**Incident ID:** {incident_id}")
with header_col2:
    st.markdown(f"**Status:** Investigating")
with header_col3:
    st.markdown(f"**Type:** Phishing")
with header_col4:
    st.markdown(f"**Timestamp:** {timestamp_utc}")

# Alert Banner
st.markdown(f"""
    <div style="background-color: #a60000; color: white; padding: 10px; border-radius: 5px; text-align: center;">
        <h2>[{alert_level.upper()}] Incident Alert</h2>
    </div>
""", unsafe_allow_html=True)

# Main Layout with 3 Columns
col1, col2, col3 = st.columns([2, 3, 2])

# -----------------------------
# Column 1: Impact Assessment
# -----------------------------
with col1:
    st.subheader("Incident Details / Impact Assessment")
    st.markdown("- **Impact Analysis**: A phishing email campaign impersonating PayPal was detected.")
    st.markdown("- **IPs**: 185.63.254.1 (Attack server, Russia)")
    st.markdown("- **Domains**: verify-paypal-login[.]com")
    st.markdown("- **Hashes**: SHA-256:5a3b2c4d1e8f (Phishing HTML attachment)")
    st.markdown("- **Registry Keys**: Not applicable for email-based phishing.")
    st.markdown("- **Targets**: Employees in Finance and HR departments.")

# -----------------------------
# Column 2: XAI Graphs
# -----------------------------
with col2:
    st.subheader("XAI Graphs")

    # Define DataFrames for each chart
    remediation_data = pd.DataFrame({
        "Action": [
            "Force password resets",
            "Block malicious domains/IPs",
            "Remove email from inboxes",
            "Conduct employee security awareness training",
            "Patch vulnerable systems"
        ],
        "Confidence Score (%)": [98, 96, 92, 85, 76],
        "Recommendation Level": [
            "Highly Recommended",
            "Highly Recommended",
            "Recommended",
            "Recommended",
            "Optional (Depends on infra)"
        ]
    })

    xai_current_data = pd.DataFrame({
        "Feature": [
            "Email sender anomaly",
            "URL reputation (malicious domain)",
            "Email body language similarity to known phishing",
            "Attachment behavior (macro execution)",
            "Login behavior anomaly"
        ],
        "Attribution Score (%)": [92, 87, 85, 82, 80]
    })

    xai_overall_data = pd.DataFrame({
        "Feature": [
            "Historical phishing domain patterns",
            "Credential theft likelihood",
            "Email metadata anomalies",
            "Known phishing template match"
        ],
        "Attribution Score (%)": [89, 88, 86, 84]
    })

    # Use session state to track which chart to show
    if "last_graph" not in st.session_state:
        st.session_state["last_graph"] = "Remediation"

    # Create three side-by-side buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("Remediation/Mitigation"):
            st.session_state["last_graph"] = "Remediation"

    with btn_col2:
        if st.button("XAI Current Attack"):
            st.session_state["last_graph"] = "Current"

    with btn_col3:
        if st.button("XAI Overall Phishing"):
            st.session_state["last_graph"] = "Overall"

    # Conditionally display the selected chart
    if st.session_state["last_graph"] == "Remediation":
        fig = px.bar(
            remediation_data,
            x="Action",
            y="Confidence Score (%)",
            color="Recommendation Level",
            title="Remediation/Mitigation Steps (Confidence Scoring)"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state["last_graph"] == "Current":
        fig = px.bar(
            xai_current_data,
            x="Feature",
            y="Attribution Score (%)",
            title="XAI Current Attack Detection Feature Attribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # "Overall"
        fig = px.bar(
            xai_overall_data,
            x="Feature",
            y="Attribution Score (%)",
            title="XAI Overall Phishing Attacks Detection Feature Attribution"
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Column 3: LLM Contextual Analysis
# -----------------------------
with col3:
    st.subheader("LLM Contextual Analysis")
    user_input = st.text_area("Type analysis context:")

    # Chat functionality
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if st.button("Send"):
        if user_input:
            # Static/hard-coded LLM response (replace with real API in production)
            response = (
                "This phishing campaign impersonates PayPal and targets Finance "
                "and HR departments. The email contains links to a spoofed login "
                "page designed to steal credentials."
            )
            st.session_state["chat_history"].append({"user": user_input, "response": response})

    # Display chat history
    for chat in st.session_state["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**LLM:** {chat['response']}")

# -----------------------------
# Bottom Panel: Logs
# -----------------------------
st.subheader("Log Snippets")
logs = """
[2023-10-05 08:15 UTC] User received phishing email: 'https://verify-paypal-login.com'
[2023-10-05 08:17 UTC] User clicked phishing link: 'https://verify-paypal-login.com/login'
[2023-10-05 08:20 UTC] Outbound connection to 185.63.254.1 on port 443
[2023-10-05 08:25 UTC] Email reported to security team.
"""
st.text_area("Logs", value=logs, height=150)
