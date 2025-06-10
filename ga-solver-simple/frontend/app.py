import streamlit as st
from streamlit_chat import message
import requests
import time
import uuid
import pandas as pd

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
CHAT_URL = f"{API_BASE_URL}/chat"
STATUS_URL = f"{API_BASE_URL}/status"
RESULTS_URL = f"{API_BASE_URL}/results"

# --- Page Setup ---
st.set_page_config(page_title="GA-TSP Solver Agent", layout="wide")
st.title("️‍🤖 GA-TSP Solver Agent")
st.write("An AI agent to help you solve the Traveling Salesperson Problem using a GPU-accelerated Genetic Algorithm.")

# --- Session State Initialization ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.task_id = None
    st.session_state.task_status = None
    st.session_state.polling = False
    # Add initial greeting from the bot
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I can help you solve TSP. To begin, just say 'solve tsp'."})

# --- Functions ---
def send_message(user_input):
    """Sends user message to the backend and updates chat."""
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    payload = {
        "session_id": st.session_state.session_id,
        "message": user_input
    }
    try:
        response = requests.post(CHAT_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        st.session_state.messages.append({"role": "assistant", "content": data["bot_response"]})
        
        if data.get("task_id"):
            st.session_state.task_id = data["task_id"]
            st.session_state.polling = True

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to communicate with the backend: {e}")

def poll_status():
    """Polls the backend for the status of an active task."""
    if not st.session_state.task_id or not st.session_state.polling:
        return

    try:
        status_response = requests.get(f"{STATUS_URL}/{st.session_state.task_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            st.session_state.task_status = status_data
            
            # Stop polling if the task is finished
            if status_data["status"] in ["completed", "failed"]:
                st.session_state.polling = False
                # Trigger a re-run to display final results
                st.rerun() 
        else:
            # Maybe the task isn't ready yet, keep polling silently
            pass
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error polling task status: {e}")
        st.session_state.polling = False

# --- UI Layout ---

# Main container
with st.container():
    # Chat history
    for i, msg in enumerate(st.session_state.messages):
        is_user = msg["role"] == "user"
        message(msg["content"], is_user=is_user, key=f"chat_msg_{i}")

    # Status placeholder
    status_placeholder = st.empty()

    # Input form
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Your message:", "", key="input")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        send_message(user_input)
        # Trigger a re-run to display the new message immediately
        st.rerun()


# --- Status and Results Display Logic ---

if st.session_state.polling:
    poll_status()
    status_data = st.session_state.task_status
    if status_data:
        status_message = f"**Task Status:** {status_data['status']} - {status_data['message']}"
        progress_details = status_data.get("details", {}).get("progress")
        if progress_details:
             status_message += f"\n> {progress_details}"
        status_placeholder.info(status_message)
    # Re-run the script to create the polling effect
    time.sleep(3)
    st.rerun()

elif st.session_state.task_id and not st.session_state.polling:
    # This block runs once polling has stopped.
    status_data = st.session_state.task_status
    if status_data and status_data["status"] == "completed":
        st.balloons()
        status_placeholder.success("🎉 Solver finished! Here is the result:")
        
        # Fetch and display the plot
        try:
            results_response = requests.get(f"{RESULTS_URL}/{st.session_state.task_id}", params={"session_id": st.session_state.session_id})
            results_response.raise_for_status()
            st.image(results_response.content, caption="Best TSP Route Found", use_column_width=True)

            # Display final stats
            details = status_data.get("details", {})
            if details:
                st.subheader("Run Statistics")
                stats_df = pd.DataFrame({
                    "Metric": ["Best Fitness (Total Cost)", "Time Taken (ms)", "Avg. Time per Generation (ms)", "Total Generations"],
                    "Value": [
                        details.get("best_fitness"), 
                        details.get("time_taken_ms"),
                        details.get("avg_time_per_gen_ms"),
                        details.get("total_generations_run")
                    ]
                })
                st.table(stats_df)

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch results: {e}")

    elif status_data and status_data["status"] == "failed":
        status_placeholder.error(f"Task Failed: {status_data['message']}")
        details = status_data.get("details", {})
        if details:
            st.text_area("Solver Log:", value=details, height=300)

    # Reset task state after showing results
    st.session_state.task_id = None
    st.session_state.task_status = None 