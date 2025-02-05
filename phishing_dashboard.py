import streamlit as st
import pandas as pd
import plotly.express as px

# Adjust Full Page Width & Sidebar Behavior
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# --------------------------
# Global CSS for uniform button style
# --------------------------
st.markdown(
    """
    <style>
    /* This will style ALL st.button elements in the app.
       They will have the same width (100% of their container),
       ensuring uniform button sizes across the three columns. */
    .stButton > button {
        width: 100% !important;
        background-color: #f8f8f8;
        border: 1px solid #ccc;
        font-size: 14px;
        padding: 6px 16px;
        border-radius: 5px;
        margin: 2px 0px;
        color: #333;
    }
    .stButton > button:hover {
        background-color: #eaeaea;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar: Incident Filter
st.sidebar.header("Incident Filter")
alert_level = st.sidebar.radio(
    "Select Alert Level:",
    ["Critical", "High", "Medium"]
)

# Determine Header Background Color
if alert_level == "Critical":
    header_color = "#a60000"  # Dark red for critical alerts
elif alert_level == "High":
    header_color = "#ff9900"  # Orange for high alerts
else:
    header_color = "#ffcc00"  # Yellow for medium alerts

# Generate Timestamp and Incident ID
incident_id = "INC-2025-000124"
timestamp_utc = "2025-02-04 10:32:45 UTC"

# Header
st.markdown(f"""
    <div style="background-color: {header_color}; color: white; padding: 3px; border-radius: 5px; text-align: center;">
        <h2> {incident_id}: [{alert_level.upper()}] Phishing Email campaign</h2>
    </div>
""", unsafe_allow_html=True)

header_col1, header_col2, header_col3= st.columns(3)
with header_col1:
    st.markdown(f"**Status:** Active (Under Investigation)")
with header_col2:
    st.markdown(f"**Target:** Finance Department (Multiple Users)")
with header_col3:
    st.markdown(f"**Timestamp:** {timestamp_utc}")

# Layout: 3 Columns
col1, col2, col3 = st.columns(3)

# -----------------------------------------------------------------------------
# Column 1: Threat Intelligence Summary and Impact Assessment
# -----------------------------------------------------------------------------
with col1:
    st.subheader("Threat Intelligence & Analysis")

    # Create toggle buttons for displaying information
    selected_view = st.radio(
        "Select View:",
        ("IoC & Impact", "Key Analysis"),
        index=0
    )

    if selected_view == "IoC & Impact":
        # Threat Intelligence Summary
        st.subheader("Threat Intelligence Summary")
        st.table(
            pd.DataFrame({
                "Label": [
                    "Hashes",
                    "Malicious Domains",
                    "Suspicious IPs",
                    "Emails",
                    "Registry Key Modifications"
                ],
                "Value": [
                    "d41d8cd98f00b204e9800998ecf8427e",
                    "secure-payments-verification[.]com",
                    "192.168.56.12 (C2 Server)",
                    "billing-alert@secure-payments-verification[.]com",
                    "HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\svchost"
                ]
            })
        )

        # Impact Assessment
        st.subheader("Impact Assessment")
        st.table(
            pd.DataFrame({
                "Label": [
                    "Users Impacted",
                    "Devices Impacted",
                    "Data Loss",
                    "Credential Theft",
                    "Expected Downtime"
                ],
                "Value": [
                    "6",
                    "3 (laptops)",
                    "Financial statements leaked (4GB exfiltrated)",
                    "2 users",
                    "3-6 hours for remediation"
                ]
            })
        )

    elif selected_view == "Key Analysis":
        st.subheader("Key Analysis")
        st.markdown("""
        - **Attack Vector:** Spear-phishing email with a malicious attachment (`invoice.pdf.exe`)
        - **Payload:** Remote Access Trojan (RAT)
        - **Campaign Name:** "Invoice Payment Reminder - Urgent Action Required"
        - **Exploited Weakness:** Lack of MFA and employees clicking on malicious links
        - **Simplified Investigation:**
            - 6 users in the finance team opened the phishing email.
            - 2 users entered credentials on a fake login page.
            - The malicious attachment executed a PowerShell script, initiating a connection to a Command-and-Control (C2) server.
            - The infected machine exfiltrated financial reports (4GB).
        """)

# -----------------------------------------------------------------------------
# Column 2: Mitigation Steps & XAI Graphs
# -----------------------------------------------------------------------------
with col2:
    st.subheader("Mitigation steps")
    # Scrollable container for steps
    st.markdown("""
    <div style="height: 250px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
        <ul>
            <li><b>Force Password Resets</b>  
            Ensure that all affected users immediately reset their passwords to prevent unauthorized access.</li>
            <li><b>Block Malicious Domains/IPs</b>  
            Update firewall and security policies to block communication with known malicious domains and IP addresses.</li>
            <li><b>Remove Email from Inboxes</b>  
            Identify and delete the phishing email from all user inboxes to prevent further engagement.</li>
            <li><b>Conduct Employee Security Awareness Training</b>  
            Provide training to employees on recognizing phishing attempts and best security practices.</li>
            <li><b>Patch Vulnerable Systems</b>  
            Ensure all software, email security gateways, and endpoint protection solutions are updated to mitigate similar threats.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("XAI Graphs")

    # DataFrames for each chart
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

    # Session state for selected chart
    if "last_graph" not in st.session_state:
        st.session_state["last_graph"] = "Remediation"

    # Uniform three-column layout for XAI buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3, gap="medium")

    with btn_col1:
        if st.button("Remediation / Mitigation"):
            st.session_state["last_graph"] = "Remediation"

    with btn_col2:
        if st.button("XAI Current Attack"):
            st.session_state["last_graph"] = "Current"

    with btn_col3:
        if st.button("XAI Overall Phishing"):
            st.session_state["last_graph"] = "Overall"

    # Display the chosen chart
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

    elif st.session_state["last_graph"] == "Overall":
        fig = px.bar(
            xai_overall_data,
            x="Feature",
            y="Attribution Score (%)",
            title="XAI Overall Phishing Attacks Detection Feature Attribution"
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Column 3: LLM Contextual Analysis
# -----------------------------------------------------------------------------
with col3:
    st.subheader("LLM Contextual Analysis")

    st.markdown(
        """
        **Suggested Questions:**
        - How can we validate if any sensitive data was exfiltrated?
        - What steps can immediately contain the threat?
        - Which users or endpoints are at greatest risk?
        - How should we improve our phishing detection mechanisms?
        """
    )

    # Add some vertical space before the text area
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Larger text area
    user_input = st.text_area("Type analysis context:", height=400)

    # Chat functionality
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if st.button("Send"):
        if user_input:
            response = (
                "This phishing campaign impersonates PayPal and targets Finance "
                "and HR departments. The email contains links to a spoofed login "
                "page designed to steal credentials."
            )
            st.session_state["chat_history"].append({"user": user_input, "response": response})

    for chat in st.session_state["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**LLM:** {chat['response']}")

# -----------------------------------------------------------------------------
# Bottom Panel: Logs
# -----------------------------------------------------------------------------
st.subheader("Incident Logs")
logs = """
[2023-10-05 08:15 UTC] User received phishing email: 'https://verify-paypal-login.com'
[2023-10-05 08:17 UTC] User clicked phishing link: 'https://verify-paypal-login.com/login'
[2023-10-05 08:20 UTC] Outbound connection to 185.63.254.1 on port 443
[2023-10-05 08:25 UTC] Email reported to security team.
"""
st.text_area("Logs", value=logs, height=150)
