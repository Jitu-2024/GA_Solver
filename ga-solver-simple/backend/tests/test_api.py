import requests
import random
import string
import time
import json

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
CHAT_URL = f"{API_BASE_URL}/chat"
STATUS_URL = f"{API_BASE_URL}/status"
# user_id is no longer needed in the request body, managed by the backend
SESSION_ID = f"session_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"

def start_solver_task(session_id: str) -> str:
    """Sends a request to start the solver and returns the task ID."""
    print("--- Sending Request to Start Solver ---")
    # Updated payload to match new ChatRequest model
    payload = {"message": "solve tsp", "session_id": session_id} 
    
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
            
            # Updated to parse the new StatusResponse model
            status = status_data.get("status")
            progress = status_data.get("progress")
            result = status_data.get("result")

            progress_str = f"{progress:.2f}%" if isinstance(progress, float) else "N/A"
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status} | Progress: {progress_str}")
            
            if status == "running":
                # The 'result' field now contains the latest log line during execution
                print(f"  -> Log: {result}")

            if status in terminal_statuses:
                print("\n--- Final Task Result ---")
                # The 'result' field contains the final detailed dictionary on completion
                print(json.dumps(result, indent=2))
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
    # For example:
    # requests.post(CHAT_URL, json={"message": "use 1000 generations", "session_id": SESSION_ID})
    
    task_id = start_solver_task(session_id=SESSION_ID)
    poll_task_status(task_id=task_id)

    print("==================================================")
    print("API test sequence completed.")

if __name__ == "__main__":
    main() 