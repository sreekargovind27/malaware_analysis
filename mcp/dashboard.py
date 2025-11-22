import sys
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# --- CRITICAL FIX: Add project root to path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Safe import
from tools import (
    scan_flow, scan_file, lookup_malware_info, get_mitigation_plan,
    check_device_history, get_model_performance, explain_prediction
)

# Load Config
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Page Setup
st.set_page_config(page_title="IoT Defense Console", page_icon="🛡️", layout="wide")
st.title("🛡️ IoT Threat Defense - Command Center")

if not api_key:
    st.error("❌ `GEMINI_API_KEY` not found. Check .env")
    st.stop()

genai.configure(api_key=api_key)

tools_list = [
    scan_flow, scan_file, lookup_malware_info, get_mitigation_plan,
    check_device_history, get_model_performance, explain_prediction
]

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔧 Operations")
    st.success(f"✅ System Online")

    uploaded_file = st.file_uploader("Upload Logs (CSV)", type="csv")
    if uploaded_file:
        # Save to a consistent temp location inside mcp folder
        path = os.path.join(current_dir, "temp.csv")
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state['csv_path'] = path
        st.success("File Ready")

# --- CHAT ---
if "history" not in st.session_state:
    model = genai.GenerativeModel('gemini-2.5-flash', tools=tools_list)
    st.session_state.chat = model.start_chat(enable_automatic_function_calling=True)
    st.session_state.history = [{"role": "model", "content": "Security Agent Online."}]

for msg in st.session_state.history:
    role = "assistant" if msg["role"] == "model" else "user"
    with st.chat_message(role):
        st.write(msg["content"])

if prompt := st.chat_input("Command"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    context = prompt
    if "csv_path" in st.session_state and "file" in prompt.lower():
        context += f"\n(Analyze file: '{st.session_state['csv_path']}')"

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                resp = st.session_state.chat.send_message(context)
                st.write(resp.text)
                st.session_state.history.append({"role": "model", "content": resp.text})
            except Exception as e:
                st.error(f"Error: {e}")
