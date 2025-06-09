import requests
import random
import string
import time
import json

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
CHAT_URL = f"{API_BASE_URL}/chat"
STATUS_URL = f"{API_BASE_URL}/status"
USER_ID = "python_async_tester"
SESSION_ID = f"session_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"

def start_solver_task(session_id: str) -> str:
    """Sends a request to start the solver and returns the task ID."""
    print("--- Sending Request to Start Solver ---")
    payload = {"user_id": USER_ID, "session_id": session_id, "message": "solve tsp"}
    
    try:
        response = requests.post(CHAT_URL, json=payload, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        print("--- Immediate Response Received ---")
        print(json.dumps(response_data, indent=2))
        
        task_id = response_data.get("task_id")
        if not task_id:
            print("!!! Error: API did not return a task_id !!!")
            exit(1)
        return task_id
    except requests.exceptions.RequestException as e:
        print(f"!!! Error starting task: {e} !!!")
        exit(1)

def poll_task_status(task_id: str):
    """Polls the status endpoint until the task is completed or fails."""
    print(f"\n--- Polling Status for Task ID: {task_id} ---")
    
    terminal_statuses = ["completed", "failed"]
    while True:
        try:
            response = requests.get(f"{STATUS_URL}/{task_id}", timeout=10)
            response.raise_for_status()
            status_data = response.json()
            
            status = status_data.get("status")
            message = status_data.get("message")
            details = status_data.get("details", {})

            print(f"[{time.strftime('%H:%M:%S')}] Status: {status} | Message: {message}")
            if details and "progress" in details:
                print(f"  -> Progress: {details['progress']}")

            if status in terminal_statuses:
                print("\n--- Final Task Details ---")
                print(json.dumps(details, indent=2))
                break
            
            time.sleep(3) # Wait 3 seconds before polling again

        except requests.exceptions.RequestException as e:
            print(f"!!! Error polling status: {e} !!!")
            break # Stop polling on error

def main():
    """Runs the main async test sequence."""
    print(f"Starting API test sequence for session: {SESSION_ID}")
    print("==================================================")
    
    # Optional: You could add preliminary calls here to set GA params if needed
    # call_chat_api(message="set population to 100", session_id=SESSION_ID)
    
    task_id = start_solver_task(session_id=SESSION_ID)
    poll_task_status(task_id=task_id)

    print("==================================================")
    print("API test sequence completed.")

if __name__ == "__main__":
    main() 