import streamlit as st
import pandas as pd
import plotly.express as px

# Adjust Full Page Width
st.set_page_config(layout="wide")

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

# Header   ########## add time stamp
st.markdown(f"""
    <div style="background-color: {header_color}; color: white; padding: 5px; border-radius: 5px; text-align: center;">
        <h2>[{alert_level.upper()}] Incident Alert</h2>
    </div>
""", unsafe_allow_html=True)

header_col1, header_col2, header_col3, header_col4 = st.columns(4)
with header_col1:
    st.markdown(f"**Incident ID:** {incident_id}")
with header_col2:
    st.markdown(f"**Status:** Active (Under Investigation)")
with header_col3:
    st.markdown(f"**Alert Type:** Phishing Email campaign") 
with header_col4:
    st.markdown(f"**Target:** Finance Department (Multiple Users)")



# Layout
col1, col2, col3 = st.columns([2, 3, 2])

# Column 1: Impact Assessment
with col1:
    st.subheader("Threat Intelligence Summary")
    st.subheader("Indicators of Compromise (IoCs):")
    st.markdown("- **Hashes:** d41d8cd98f00b204e9800998ecf8427e")
    st.markdown("- **Malicious Domains:** secure-payments-verification[.]com")
    st.markdown("- **Suspicious IPs:** 192.168.56.12 (C2 Server)")
    st.markdown("- **Emails:** billing-alert@secure-payments-verification[.]com")
    st.markdown("- **Registry Key Modifications:** HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\svchost")


# Column 2: XAI Graphs
with col2:
    st.subheader("XAI Graphs")
    # st.write("Mitigation steps for admin and users:")
    # graph_data = pd.DataFrame({
    #     "Step": ["Report phishing email", "Block malicious IP", "Monitor suspicious activity"],
    #     "Effectiveness": [95, 88, 80]
    # })
    # fig = px.bar(graph_data, x="Step", y="Effectiveness", title="Mitigation Effectiveness")
    # st.plotly_chart(fig, use_container_width=True)

    # # Additional Graph: Affected Departments
    # affected_data = pd.DataFrame({
    #     "Department": ["Finance", "HR", "IT", "Marketing"],
    #     "Affected Users": [12, 8, 4, 2]
    # })
    # fig2 = px.pie(affected_data, values="Affected Users", names="Department", title="Affected Users by Department")
    # st.plotly_chart(fig2, use_container_width=True)

# Column 3: LLM Contextual Analysis
with col3:
    st.subheader("LLM Contextual Analysis")
    user_input = st.text_area("Type analysis context:")

    # Chat functionality
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if st.button("Send"):
        if user_input:
            response = "This phishing campaign impersonates PayPal and targets Finance and HR departments. The email contains links to a spoofed login page designed to steal credentials."
            st.session_state["chat_history"].append({"user": user_input, "response": response})

    # Display chat history
    for chat in st.session_state["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**LLM:** {chat['response']}")

# Bottom Panel: Logs
st.subheader("Log Snippets")
logs = """
[2023-10-05 08:15 UTC] User received phishing email: 'https://verify-paypal-login.com'
[2023-10-05 08:17 UTC] User clicked phishing link: 'https://verify-paypal-login.com/login'
[2023-10-05 08:20 UTC] Outbound connection to 185.63.254.1 on port 443
[2023-10-05 08:25 UTC] Email reported to security team.
"""
st.text_area("Logs", value=logs, height=150)
