from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
import uvicorn
from typing import List, Dict, Any, Tuple

# Add the parent directory of 'backend' to sys.path to allow importing ga_solver_chatbot
# This assumes 'backend' is a subdirectory of 'ga-solver-simple'
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir) # No longer needed for tools.ga_solver

# Now we can import the solver function
from tools.ga_solver import run_gpu_ga_tsp_solver

app = FastAPI(
    title="GA Solver Chatbot API",
    description="API for interacting with the TSP Genetic Algorithm Solver Chatbot.",
    version="0.1.0"
)

# In-memory store for session data
# Structure: {(user_id, session_id): {"history": [], "ga_params": {}, "tsp_data": {}}}
session_store: Dict[Tuple[str, str], Dict[str, Any]] = {}

class ChatMessage(BaseModel):
    user_id: str = "default_user"
    session_id: str # Made session_id mandatory for session tracking
    message: str

class ChatResponse(BaseModel):
    bot_response: str
    session_id: str
    # We can add more fields later, like suggested_actions, solver_status, etc.

def get_session_data(user_id: str, session_id: str) -> Dict[str, Any]:
    """Retrieves or initializes session data."""
    session_key = (user_id, session_id)
    if session_key not in session_store:
        session_store[session_key] = {
            "history": [],  # List of {"user": "message"} or {"bot": "message"}
            "ga_params": { # To store extracted GA parameters
                "population_size": 1000, # Default example
                "num_generations": 50000, # Default example
                "tournament_size": 10,
                "mutation_rate": 0.1,
                "elitism_rate": 0.05,
            },
            "tsp_data": { # To store TSP problem data (coordinates or file path)
                "coordinates": None,
                "tsp_data_file": None,
                "initial_genomes_file": None
            }
        }
    return session_store[session_key]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatMessage):
    """
    Receives a user message, manages session state, and returns a bot response.
    """
    print(f"Received message from {payload.user_id} in session {payload.session_id}: {payload.message}")
    
    session_data = get_session_data(payload.user_id, payload.session_id)
    session_data["history"].append({"user": payload.message})
    
    bot_reply = f"Acknowledged: '{payload.message}'. Session history is being tracked."

    # --- LLM and Solver Logic (Illustrative with Session Data) ---
    # This section will be significantly expanded with actual LLM integration.
    # For now, it demonstrates using and updating session_data.

    user_intent = "unknown" # This would be determined by an LLM

    # Simplified intent detection (to be replaced by LLM)
    if "solve tsp" in payload.message.lower():
        user_intent = "solve_tsp"
    elif "hello" in payload.message.lower() or "hi" in payload.message.lower():
        user_intent = "greeting"
    elif "set population to" in payload.message.lower():
        user_intent = "set_ga_param"
        try:
            # Extremely basic parsing, LLM would handle this robustly
            pop_size = int(payload.message.split("set population to")[1].strip())
            session_data["ga_params"]["population_size"] = pop_size
            bot_reply = f"Okay, population size set to {pop_size} for session {payload.session_id}."
        except ValueError:
            bot_reply = "Sorry, I couldn't understand the population size value."
    elif "what is population size" in payload.message.lower():
        user_intent = "query_ga_param"
        pop_size = session_data["ga_params"].get("population_size", "not set")
        bot_reply = f"Current population size for this session is: {pop_size}."


    if user_intent == "solve_tsp":
        # Try to use data from session. LLM would confirm/clarify if data is missing/ambiguous.
        ga_params = session_data["ga_params"]
        tsp_data = session_data["tsp_data"]

        # Mocking coordinates if not present, LLM would prompt user for this
        if not tsp_data.get("coordinates") and not tsp_data.get("tsp_data_file"):
            mock_coordinates = [(10.0, 10.0), (20.0, 20.0), (30.0, 10.0)]
            print(f"Mocking coordinates for solver call: {mock_coordinates}")
            tsp_data["coordinates"] = mock_coordinates # Save to session for future
            bot_reply += f" Using mock coordinates {mock_coordinates}."

        print(f"Attempting to call run_gpu_ga_tsp_solver with params from session: {ga_params}, data: {tsp_data}")
        
        try:
            solver_results = run_gpu_ga_tsp_solver(
                tsp_data_file=tsp_data.get("tsp_data_file"),
                coordinates=tsp_data.get("coordinates"),
                initial_genomes_file=tsp_data.get("initial_genomes_file"),
                population_size=ga_params.get("population_size", 1000),
                num_generations=ga_params.get("num_generations", 50000),
                tournament_size=ga_params.get("tournament_size", 10),
                mutation_rate=ga_params.get("mutation_rate", 0.1),
                elitism_rate=ga_params.get("elitism_rate", 0.05),
                cuda_solver_executable_path="./ga_solver_executable"
            )
            print(f"Solver results: {solver_results}")
            bot_reply = f"Solver interaction attempted. Status: {solver_results.get('status')}, Message: {solver_results.get('message')}"
        except Exception as e:
            print(f"Error during solver call: {e}")
            bot_reply = f"Error during solver interaction: {e}"

    elif user_intent == "greeting":
        bot_reply = f"Hello {payload.user_id}! How can I help you with TSP today? (Session: {payload.session_id})"
    
    # If bot_reply wasn't set by specific intent logic, use the default acknowledgement
    elif user_intent == "unknown" and bot_reply.startswith("Acknowledged"):
         bot_reply = f"I received: '{payload.message}'. I can help with TSP. You can ask me to solve, set parameters, or ask about them. (Session: {payload.session_id})"


    session_data["history"].append({"bot": bot_reply})
    # print(f"Session store updated for {payload.user_id}/{payload.session_id}: {session_store[(payload.user_id, payload.session_id)]}")

    return ChatResponse(bot_response=bot_reply, session_id=payload.session_id)

# To run this app:
# 1. Ensure you are in the 'ga-solver-simple/backend' directory.
# 2. Install dependencies: pip install -r requirements.txt
# 3. Run Uvicorn: uvicorn main:app --reload
#    You should see output like: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
# 4. You can then send POST requests to http://127.0.0.1:8000/chat
#    Example using curl (ensure you generate a unique session_id for each new conversation flow):
#    curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "test_user", "session_id": "session_xyz123", "message": "Hello TSP Solver!"}'
#    curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "test_user", "session_id": "session_xyz123", "message": "set population to 100"}'
#    curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "test_user", "session_id": "session_xyz123", "message": "what is population size"}'
#    curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "test_user", "session_id": "session_xyz123", "message": "solve tsp"}'

if __name__ == "__main__":
    # This block allows running the app directly with python main.py for simple testing,
    # but uvicorn is preferred for development and production.
    
    print("Starting Uvicorn server. Navigate to http://127.0.0.1:8000/docs for API documentation.")
    uvicorn.run(app, host="127.0.0.1", port=8000) 