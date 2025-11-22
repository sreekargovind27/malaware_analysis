import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
import random
import time
from dotenv import load_dotenv

# --- CONFIGURATION ---
st.set_page_config(page_title="IoT Defense Console", page_icon="🛡️", layout="wide")
load_dotenv()


# --- DUMMY TOOLS (DEFINED LOCALLY TO PREVENT CRASHES) ---
# These simulate your backend logic perfectly for the demo.

def scan_flow(duration: float, bytes_t: int, port: int) -> str:
    """Analyzes a single network flow for threats."""
    # Simulation Logic
    score = 0.1
    if port in [23, 2323, 80, 8080]: score += 0.4
    if bytes_t > 5000: score += 0.3

    is_malicious = score > 0.5
    family = "Benign"

    if is_malicious:
        if port == 23:
            family = "Mirai"
        elif bytes_t > 50000:
            family = "Torii"
        else:
            family = "Gagfyt"

    # Return JSON string for Gemini
    return str({
        "status": "MALICIOUS" if is_malicious else "BENIGN",
        "confidence": 0.98 if is_malicious else 0.92,
        "primary_model": "XGBoost (Attack Type)",
        "detected_family": family,
        "risk_score": int(score * 100)
    })


def scan_file(file_path: str) -> str:
    """Analyses a CSV log file."""
    # Simulate processing a file path
    return (f"Batch Analysis Result for '{file_path}':\n"
            "- Total Flows: 12,405\n"
            "- Malicious Detected: 842\n"
            "- Top Threat: Mirai (Port 23)\n"
            "- Status: CRITICAL")


def lookup_malware_info(malware_name: str) -> str:
    """Retrieves details about a malware family."""
    name = malware_name.lower()
    if "mirai" in name:
        return "📚 Mirai: IoT botnet targeting Telnet (Port 23). Uses default credentials."
    if "torii" in name:
        return "📚 Torii: Sophisticated botnet focused on persistence and data exfiltration."
    return "No specific intelligence found in local database."


def get_mitigation_plan(threat_name: str) -> str:
    """Provides actionable steps to stop a specific threat."""
    return "🛡️ PLAN: 1. Block Port 23/2323. 2. Reboot device to clear RAM. 3. Rotate credentials."


def check_device_history(ip_address: str) -> str:
    """Checks reputation of an IP."""
    status = random.choice(["Clean", "Suspicious", "Known Botnet"])
    return f"🔎 REPORT [{ip_address}]: {status} (Seen {random.randint(1, 50)} times)"


def get_model_performance() -> str:
    """Returns model metrics."""
    return """
    📊 Model Registry Status:
    1. XGBoost (Attack Type): Accuracy 97.89% ✅
    2. LightGBM (Malware Family): Accuracy 92.15% ✅
    3. Graph Neural Network: AUC 0.91 ✅
    """


def explain_prediction(flow_id: str) -> str:
    """Explains the AI decision."""
    return "💡 EXPLAINER: High risk score driven by destination port 23 and high upload ratio."


# --- MAIN APP UI ---

st.title("🛡️ IoT Threat Defense - Command Center")

# Sidebar Setup
with st.sidebar:
    st.header("🔧 Operations")
    st.success(f"✅ System Online\n- Mode: Demo / Simulation")

    # API Key Input (Fallback if .env fails)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key", type="password")

    st.divider()

    # File Uploader
    uploaded_file = st.file_uploader("Upload Logs (CSV)", type="csv")
    if uploaded_file:
        # Save to a simple temp file in current dir
        with open("temp_upload.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state['csv_path'] = "temp_upload.csv"
        st.success("File Ready")

# Gemini Setup
if not api_key:
    st.warning("⚠️ Please provide a Gemini API Key to start the agent.")
    st.stop()

try:
    genai.configure(api_key=api_key)

    # Register the LOCAL dummy functions
    tools_list = [
        scan_flow, scan_file, lookup_malware_info, get_mitigation_plan,
        check_device_history, get_model_performance, explain_prediction
    ]

    if "history" not in st.session_state:
        model = genai.GenerativeModel('gemini-2.5-flash', tools=tools_list)
        st.session_state.chat = model.start_chat(enable_automatic_function_calling=True)
        st.session_state.history = [{"role": "model", "content": "Security Agent Online. Ready to scan flows."}]

except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

# Chat Interface
for msg in st.session_state.history:
    role = "assistant" if msg["role"] == "model" else "user"
    with st.chat_message(role):
        st.write(msg["content"])

if prompt := st.chat_input("Try: 'Scan flow on port 23' or 'Check system stats'"):
    # User Message
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Context Injection
    context = prompt
    if "csv_path" in st.session_state and "file" in prompt.lower():
        context += f"\n(System: The user uploaded a file at '{st.session_state['csv_path']}')"

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                resp = st.session_state.chat.send_message(context)
                st.write(resp.text)
                st.session_state.history.append({"role": "model", "content": resp.text})
            except Exception as e:
                st.error(f"Error: {e}")