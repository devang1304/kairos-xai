import streamlit as st
import pandas as pd
import plotly.express as px
import openai

# Adjust Full Page Width & Sidebar Behavior
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# --------------------------
# Global CSS for uniform button style
# --------------------------
st.markdown(
    """
    <style>
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

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTable {
        margin-top: -10px;
        margin-bottom: -30px;
    }
    h4 {
        margin-top: 5px;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
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

header_col1, header_col2, header_col3 = st.columns(3)

with header_col1:
    st.markdown(f"""
        <div style="font-size:18px; font-weight:bold;">
            <span style="color:#ff4d4d;">Status:</span> Active (Under Investigation)
        </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown(f"""
        <div style="font-size:18px; font-weight:bold;">
            <span style="color:#3399ff;">Target:</span> Finance Department (Multiple Users)
        </div>
    """, unsafe_allow_html=True)

with header_col3:
    st.markdown(f"""
        <div style="font-size:18px; font-weight:bold;">
            <span style="color:#33cc33;">Timestamp:</span> {timestamp_utc}
        </div>
    """, unsafe_allow_html=True)


# Body Layout: 3 Columns
col1, col2, col3 = st.columns(3)

# -----------------------------------------------------------------------------
# Column 1: Threat Intelligence Summary and Impact Assessment
# -----------------------------------------------------------------------------
with col1:
    # Indicators of Compromise
    st.markdown("<h4 style='margin-bottom: 5px;'>Indicators of Compromise (IoC)</h4>",
                unsafe_allow_html=True)
    st.table(
        pd.DataFrame({
            "Label": [
                "Hashes",
                "Domains",
                "IPs",
                "Emails",
            ],
            "Value": [
                "d41d8cd98f00b204e9800998ecf8427e",
                "secure-payments-verification[.]com",
                "192.168.56.12 (C2 Server)",
                "billing-alert@secure-payments-verification[.]com",
            ]
        })
    )

    # Impact Assessment
    st.markdown("<h4 style='margin-top: 5px; margin-bottom: 5px;'>Impact Assessment</h4>",
                unsafe_allow_html=True)
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

    # Key Analysis
    st.markdown("<h4 style='margin-top: 5px; margin-bottom: 5px;'>Key Analysis</h4>",
                unsafe_allow_html=True)
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
    st.markdown("<h4 style='margin-top: 5px; margin-bottom: 5px;'>Mitigation Steps</h4>",
                unsafe_allow_html=True)

    # Create a scrollable container with a fixed height
    st.markdown("""
        <ol>
            <li>Isolate Impacted Devices  
            <li>Disable Compromised Accounts & Reset Passwords  
            <li>Block IoCs at Network & Endpoint Levels  
            <li>Notify Affected Users & Security Operations Center (SOC)  
            <li>Enable MFA on All High-Risk Accounts  
        </ol>
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='margin-top: 5px; margin-bottom: 5px;'>Explainable AI Graphs</h4>",
                unsafe_allow_html=True)

    # DataFrames for each chart
    remediation_data = pd.DataFrame({
        "Action": [
            "Force password<br>resets",
            "Block malicious<br>domains/IPs",
            "Remove email<br>from inboxes",
            "Conduct employee security<br>awareness training",
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
            "Email sender<br>anomaly",
            "URL reputation<br>(malicious domain)",
            "Email body<br>language similarity<br>to known phishing",
            "Attachment behavior<br>(macro execution)",
            "Login behavior<br>anomaly"
        ],
        "Attribution Score (%)": [92, 87, 85, 82, 80]
    })

    # Remediation/Mitigation Steps Graph
    fig = px.bar(
        remediation_data,
        x="Action",
        y="Confidence Score (%)",
        color="Recommendation Level",
        title="Remediation/Mitigation Steps (Confidence Scoring)"
    )
    fig.update_layout(
        # Center title, smaller size
        title=dict(font=dict(size=14), x=0.5, y=0.95),
        margin=dict(l=5, r=5, t=30, b=50),  # Compact margins
        legend=dict(
            orientation="v",  # Vertical legend
            y=0.5,  # Center legend vertically
            x=1.02,  # Position legend outside the plot on the right
            xanchor="left",
            font=dict(size=12)
        ),
        xaxis=dict(
            tickangle=-45,  # Rotate labels diagonally
            tickfont=dict(size=12),
            automargin=True
        ),
        yaxis=dict(
            title=dict(font=dict(size=12)),
            tickfont=dict(size=12),
            range=[70, 100]  # Dynamic scaling for y-axis
        ),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    # XAI Current Attack Detection Feature Attribution Graph
    fig = px.bar(
        xai_current_data,
        x="Feature",
        y="Attribution Score (%)",
        title="XAI Current Attack Detection Feature Attribution"
    )
    fig.update_layout(
        # Center title, smaller size
        title=dict(font=dict(size=14), x=0.5, y=0.95),
        margin=dict(l=5, r=5, t=30, b=50),  # Increased bottom margin
        legend=dict(
            orientation="v",  # Vertical legend
            y=-0.3,  # Push legend below the chart
            x=0.5,  # Center legend
            xanchor="center",
            font=dict(size=12)
        ),
        xaxis=dict(
            tickangle=-45,  # Rotate labels diagonally
            tickfont=dict(size=10),
            automargin=True
        ),
        yaxis=dict(
            title=dict(font=dict(size=12)),
            tickfont=dict(size=12),
            range=[75, 100]  # Dynamic scaling for y-axis
        ),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Column 3: LLM Contextual Analysis
# -----------------------------------------------------------------------------
with col3:
    st.markdown("<h4 style='margin-top: 5px; margin-bottom: 5px;'>LLM Contextual Analysis</h4>",
                unsafe_allow_html=True)

    # Chat history storage
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "user": "contextual analysis of the incident",
                "response": "<br><b>- Previous similar phishing attack:</b> Finance_Dept_Phish_2024-12-10<br><b>- Common Indicators:</b> Email template matches 84% with prior attacks.<br><b>- Similar past target:</b> HR & Finance users.<br><b>- Attack trend:</b> Increasing sophistication in social engineering tactics."
            }
        ]

    # Input box and send button side by side
    input_col, button_col = st.columns([8, 2])
    with input_col:
        user_input = st.text_input("user_input", label_visibility="collapsed")
    with button_col:
        if st.button("Send"):
            if user_input:
                # Example response from the LLM
                response = ("<br>The estimated financial impact includes direct losses from unauthorized transactions, potential regulatory fines, and remediation costs. Additionally, reputational damage and lost productivity contribute to overall financial strain. A detailed cost breakdown is available upon request."
                )
                st.session_state["chat_history"].append(
                    {"user": user_input, "response": response})

            # Clear the input field
            st.session_state["user_input"] = ""

    for chat in st.session_state["chat_history"]:
        # User message bubble
        st.markdown(f"""
        <div style="text-align: right; margin-bottom: 10px;">
            <div style="display: inline-block; background-color: #d1f2eb; padding: 8px 12px; border-radius: 15px; max-width: 80%; word-wrap: break-word; color: black;">
                <b>You:</b> {chat['user']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # LLM response bubble
        st.markdown(f"""
        <div style="text-align: left; margin-bottom: 10px;">
            <div style="display: inline-block; background-color: #f2f2f2; padding: 8px 12px; border-radius: 15px; max-width: 80%; word-wrap: break-word; color: black;">
                <b>LLM:</b> {chat['response']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


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
