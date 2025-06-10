from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sys
import os
import uvicorn
from typing import List, Dict, Any, Tuple, Optional
import uuid
import io
import matplotlib.pyplot as plt
from enum import Enum
import json
import openai

# --- LLM Configuration ---
# IMPORTANT: Set your OpenAI API key as an environment variable
# export OPENAI_API_KEY='your-key-here'
try:
    client = openai.OpenAI()
    # Test the client connection
    client.models.list()
except openai.APIKeyNotSetError:
    print("FATAL ERROR: The OPENAI_API_KEY environment variable is not set.")
    print("Please set it before running the application.")
    # Use a placeholder client to avoid crashing on import, but it will fail on use
    client = None
except Exception as e:
    print(f"An error occurred initializing the OpenAI client: {e}")
    client = None


# --- App Setup ---
sys.path.append(os.path.join(os.path.dirname(__file__), "tools"))

from tools.ga_solver import run_gpu_ga_solver_background, find_default_solver_path
# Read the docstring to use as context for the LLM
ga_solver_docstring = run_gpu_ga_solver_background.__doc__


app = FastAPI(
    title="GA Solver Multi-Agent API",
    description="An agentic API using distinct, state-driven agents to solve TSP with a GPU-accelerated Genetic Algorithm.",
    version="1.2.0"
)

# --- State Management ---
session_store: Dict[str, Dict[str, Any]] = {}
task_store: Dict[str, Dict[str, Any]] = {}

class AgentState(str, Enum):
    IDLE = "IDLE" # New initial state
    GATHERING_PARAMETERS = "GATHERING_PARAMETERS"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    EXECUTING_TASK = "EXECUTING_TASK"
    PROVIDING_FEEDBACK = "PROVIDING_FEEDBACK"

class ChatMessage(BaseModel):
    user_id: str = "default_user"
    session_id: str
    message: str

class ChatResponse(BaseModel):
    bot_response: str
    session_id: str
    task_id: Optional[str] = None
    state: AgentState

class TaskStatus(BaseModel):
    status: str
    message: str
    details: Optional[Any] = None

# --- Session Handling ---

def get_session(session_id: str) -> Dict[str, Any]:
    """Retrieves or initializes a user session."""
    if session_id not in session_store:
        session_store[session_id] = {
            "history": [], # To store conversation history for the LLM
            "state": AgentState.IDLE,
            "ga_params": {
                "population_size": 1000,
                "num_generations": 50000,
                "tournament_size": 10,
                "mutation_rate": 0.1,
                "elitism_rate": 0.05,
            },
            "tsp_data": {
                "coordinates": None,
                "tsp_data_file": None,
            }
        }
    return session_store[session_id]

# --- Core Agent & LLM Logic ---

def call_llm(system_prompt: str, user_message: str) -> Optional[Dict[str, Any]]:
    """Generic function to call the LLM with a specific system prompt."""
    if not client:
        return {"error": "LLM client not initialized."}
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {"error": str(e)}

# --- State-Specific Agents ---

def run_idle_agent(session: Dict[str, Any], user_message: str) -> Tuple[str, AgentState]:
    """Agent that classifies the user's initial intent."""
    system_prompt = """
You are a classification agent. Your job is to determine if the user wants to solve a Traveling Salesperson Problem (TSP).
The user might also just be greeting you.
Respond with a JSON object with one key, "intent", which can be one of three values: "SOLVE_TSP", "GREETING", or "UNKNOWN".
If you identify an intent to solve a TSP, also extract any city `coordinates` if they are provided in the message.
Example: "I need to solve a TSP for these locations: [(10,20), (30,40)]" -> {"intent": "SOLVE_TSP", "coordinates": [[10,20], [30,40]]}
"""
    response = call_llm(system_prompt, user_message)
    intent = response.get("intent", "UNKNOWN")

    if response.get("coordinates"):
        session["tsp_data"]["coordinates"] = response["coordinates"]

    if intent == "SOLVE_TSP":
        if session["tsp_data"]["coordinates"]:
            return (f"Understood. I've received these coordinates: {session['tsp_data']['coordinates']}. "
                    "I'm ready to run the solver with default parameters. Shall I proceed?", AgentState.AWAITING_CONFIRMATION)
        return ("Understood. I can help solve a TSP. Please provide the city coordinates to get started.", AgentState.GATHERING_PARAMETERS)
    elif intent == "GREETING":
        return ("Hello! I'm an AI agent designed to solve the Traveling Salesperson Problem. How can I help you today?", AgentState.IDLE)
    else:
        return ("I'm not sure how to help with that. My purpose is to solve TSP problems. You can start by telling me you want to 'solve a tsp'.", AgentState.IDLE)


def run_parameter_gathering_agent(session: Dict[str, Any], user_message: str) -> Tuple[str, AgentState]:
    """Agent that extracts TSP parameters from user messages."""
    system_prompt = f"""
You are a parameter extraction agent for a TSP solver tool.
Your goal is to extract any parameters the user provides for the solver.
You MUST refer to the tool's documentation to understand what to look for.

--- TOOL DOCUMENTATION ---
{ga_solver_docstring}
--- END TOOL DOCUMENTATION ---

The user said: "{user_message}"
Extract any arguments (like `coordinates`, `population_size`, `num_generations`, etc.) you find into a JSON object.
If no parameters are found, return an empty JSON object.
Example: "Use these coords: [(10,20)] and a population of 500" -> {"coordinates": [[10,20]], "population_size": 500}
"""
    response = call_llm(system_prompt, user_message)
    
    if not response or response.get("error"):
        return "There was an issue processing your request with the LLM.", AgentState.GATHERING_PARAMETERS

    updated_params = False
    for key, value in response.items():
        if key in session["ga_params"]:
            session["ga_params"][key] = value
            updated_params = True
        elif key in session["tsp_data"]:
            session["tsp_data"][key] = value
            updated_params = True

    if not updated_params:
        return "I didn't recognize any valid parameters in your message. Please provide coordinates or specify a parameter to set, like 'set population size to 1500'.", AgentState.GATHERING_PARAMETERS

    if session["tsp_data"].get("coordinates"):
        return f"Parameters updated. The current coordinates are {session['tsp_data']['coordinates']}. Ready to proceed?", AgentState.AWAITING_CONFIRMATION
    else:
        return "Parameters updated. I still need city coordinates to proceed.", AgentState.GATHERING_PARAMETERS


def run_confirmation_agent(session: Dict[str, Any], user_message: str) -> Tuple[str, AgentState]:
    """Agent that determines if the user is confirming or denying an action."""
    system_prompt = """
You are a confirmation agent. Your job is to determine if the user's message is a "yes" or a "no".
Respond with a JSON object with one key, "intent", which can be "CONFIRM" or "REJECT".
"""
    response = call_llm(system_prompt, user_message)
    intent = response.get("intent", "REJECT")

    if intent == "CONFIRM":
        if not session["tsp_data"].get("coordinates") and not session["tsp_data"].get("tsp_data_file"):
            return "I can't proceed without city data. Please provide coordinates.", AgentState.GATHERING_PARAMETERS
        # This is where we trigger the background task
        return "CONFIRMED", AgentState.EXECUTING_TASK # Special message to trigger execution
    else: # REJECT
        return "Okay, I will not run the solver. What would you like to change?", AgentState.GATHERING_PARAMETERS

# --- Main Agent Router ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatMessage, background_tasks: BackgroundTasks):
    """Receives a user message and orchestrates the agent's response."""
    print(f"Received message from session {payload.session_id}: {payload.message}")
    
    session = get_session(payload.session_id)
    session["history"].append({"role": "user", "content": payload.message})

    state = session["state"]
    bot_reply = "An unexpected error occurred."
    task_id = None

    # Route to the appropriate agent based on the current state
    if state == AgentState.IDLE:
        bot_reply, new_state = run_idle_agent(session, payload.message)
    elif state == AgentState.GATHERING_PARAMETERS:
        bot_reply, new_state = run_parameter_gathering_agent(session, payload.message)
    elif state == AgentState.AWAITING_CONFIRMATION:
        bot_reply, new_state = run_confirmation_agent(session, payload.message)
    else:
        bot_reply, new_state = "I'm in an unhandled state.", state

    session["state"] = new_state
    
    # Handle the special case for starting the execution
    if bot_reply == "CONFIRMED":
        task_id = str(uuid.uuid4())
        task_store[task_id] = {"status": "pending", "message": "Task is queued."}
        
        background_tasks.add_task(
            run_gpu_ga_solver_background,
            task_store=task_store,
            task_id=task_id,
            coordinates=session["tsp_data"].get("coordinates"),
            **session["ga_params"]
        )
        bot_reply = f"Excellent. The solver is now running. Your task ID is: {task_id}."
        session["state"] = AgentState.PROVIDING_FEEDBACK # Transition after starting

    session["history"].append({"role": "assistant", "content": bot_reply})
    
    print(f"Session {payload.session_id} new state: {session['state']}")
    return ChatResponse(bot_response=bot_reply, session_id=payload.session_id, task_id=task_id, state=session['state'])

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Polls for the status of a running solver job."""
    status = task_store.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@app.get("/results/{task_id}")
async def get_task_results(task_id: str, session_id: str):
    """Retrieves the final result of a task, which could be a plot or an error."""
    task = task_store.get(task_id)
    session = get_session(session_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not complete. Current status: {task['status']}")

    details = task.get("details", {})
    route = details.get("best_route")
    coordinates = session.get("tsp_data", {}).get("coordinates")

    if not route or not coordinates:
        raise HTTPException(status_code=500, detail="Result data is incomplete and cannot be plotted.")

    # Create a copy to avoid modifying the stored result
    plot_route = route[:]
    plot_route.append(plot_route[0])
    
    # Unzip coordinates for plotting
    x_coords, y_coords = zip(*[coordinates[i] for i in plot_route])

    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, 'o-', label='TSP Route')
    plt.scatter([c[0] for c in coordinates], [c[1] for c in coordinates], c='red', zorder=5)
    
    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, f' {i}', fontsize=12)

    plt.title("TSP Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")

# --- Server Startup ---

if __name__ == "__main__":
    if not client:
        print("\nHalting server startup due to OpenAI client initialization failure.")
    else:
        print("Starting Uvicorn server for GA Solver Multi-Agent API.")
        print("Navigate to http://127.0.0.1:8000/docs for API documentation.")
        uvicorn.run(app, host="127.0.0.1", port=8000) 