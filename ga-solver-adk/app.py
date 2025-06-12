import streamlit as st
from streamlit_chat import message
import requests
import time
import uuid
import pandas as pd

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8888"
CHAT_URL = f"{API_BASE_URL}/chat"
RESPOND_URL = f"{API_BASE_URL}/respond" # New endpoint for human-in-the-loop
STATUS_URL = f"{API_BASE_URL}/status"
RESULTS_URL = f"{API_BASE_URL}/results"

# --- Page Setup ---
st.set_page_config(page_title="ADK GA-TSP Solver", layout="wide")
st.title("🤖 ADK-Powered GA-TSP Solver Agent")
st.write("A multi-agent system to help you define and solve optimization problems.")

# --- Session State Initialization ---
if 'session_id' not in st.session_state:
    print("making stuff")
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.user_id = f"user_{str(uuid.uuid4())[:8]}"
    st.session_state.messages = []
    st.session_state.task_id = None
    st.session_state.task_status = None
    st.session_state.polling = False
    st.session_state.needs_human_input = False # New state to manage the human-in-the-loop flow
    # Add initial greeting from the bot
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm an AI assistant specializing in optimization. What problem can I help you solve today?"})

# --- Functions for API Communication ---
def handle_agent_response(data: dict):
    """Processes the response from the backend and updates session state."""
    st.session_state.messages.append({"role": "assistant", "content": data["bot_response"]})
    
    if data.get("needs_human_input"):
        st.session_state.needs_human_input = True
    else:
        st.session_state.needs_human_input = False
    
    if data.get("task_id"):
        st.session_state.task_id = data["task_id"]
        st.session_state.polling = True

def send_message(user_input: str, endpoint: str):
    """Sends a message to the specified backend endpoint."""
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    payload = {
        "session_id": st.session_state.session_id,
        "user_id": st.session_state.user_id,
        "message": user_input
    }
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        handle_agent_response(data)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to communicate with the backend: {e}")
        st.session_state.needs_human_input = False

def poll_status():
    """Polls the backend for the status of an active task."""
    if not st.session_state.task_id or not st.session_state.polling:
        return
    try:
        status_response = requests.get(f"{STATUS_URL}/{st.session_state.task_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            st.session_state.task_status = status_data
            if status_data["status"] in ["completed", "failed"]:
                st.session_state.polling = False
                st.rerun() 
        else:
            pass
    except requests.exceptions.RequestException as e:
        st.error(f"Error polling task status: {e}")
        st.session_state.polling = False

# --- UI Layout ---
chat_container = st.container()
input_container = st.container()

with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        is_user = msg["role"] == "user"
        message(msg["content"], is_user=is_user, key=f"chat_msg_{i}")

status_placeholder = st.empty()

# --- Main Interaction Logic ---
if st.session_state.needs_human_input:
    # Human-in-the-loop mode
    with input_container:
        st.info("The agent is paused and requires your input to continue.")
        with st.form(key='human_response_form', clear_on_submit=True):
            human_response = st.text_input("Your Response:", "", key="human_input")
            submitted = st.form_submit_button("Send Response")

        if submitted and human_response:
            send_message(human_response, RESPOND_URL)
            st.rerun()
else:
    # Normal chat mode
    with input_container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Your message:", "", key="input")
            submitted = st.form_submit_button("Send")

        if submitted and user_input:
            send_message(user_input, CHAT_URL)
            st.rerun()

# --- Status and Results Display Logic ---
if st.session_state.polling:
    poll_status()
    status_data = st.session_state.task_status
    if status_data:
        status_message = f"**Task Status:** {status_data['status']} - {status_data['message']}"
        # Forcibly rewrite the logic to be robust and break any cache.
        if (details := status_data.get("details")) and (progress := details.get("progress")):
            status_message += f"\n> {progress}"
        status_placeholder.info(status_message)
    time.sleep(3)
    st.rerun()

elif st.session_state.task_id and not st.session_state.polling:
    # This block runs once polling has stopped.
    status_data = st.session_state.task_status
    if status_data and status_data["status"] == "completed":
        st.balloons()
        status_placeholder.success("🎉 Solver finished! Here is the result:")
        
        try:
            results_response = requests.get(f"{RESULTS_URL}/{st.session_state.task_id}", params={"session_id": st.session_state.session_id})
            results_response.raise_for_status()
            st.image(results_response.content, caption="Best TSP Route Found", use_column_width=True)

            details = status_data.get("details", {})
            if details:
                st.subheader("Run Statistics")
                stats_df = pd.DataFrame({
                    "Metric": ["Best Fitness (Total Cost)", "Time Taken (ms)", "Avg. Time per Generation (ms)", "Total Generations"],
                    "Value": [details.get("best_fitness"), details.get("time_taken_ms"), details.get("avg_time_per_gen_ms"), details.get("total_generations_run")]
                })
                st.table(stats_df)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch results: {e}")
    elif status_data and status_data["status"] == "failed":
        status_placeholder.error(f"Task Failed: {status_data['message']}")
        # ... error details display ...
    
    # Reset task state after showing results
    st.session_state.task_id = None
    st.session_state.task_status = None 