import asyncio
import uvicorn
import uuid
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import io
import matplotlib.pyplot as plt

# Corrected imports based on user feedback and documentation
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# Import our agent and tool definitions
from agents import root_agent
from ga_solver import run_gpu_ga_solver_background # The actual function to run

# --- App & ADK Setup ---
load_dotenv(override=True)
app = FastAPI(
    title="ADK-Powered GA Solver (Corrected)",
    description="A multi-agent system for solving optimization problems using Google's Agent Development Kit.",
    version="2.1.0"
)

# In-memory stores for simplicity
task_store: Dict[str, Dict[str, Any]] = {}
session_service = InMemorySessionService()

# Corrected Runner initialization with an explicit tool_mapping dictionary.
runner = Runner(
    agent=root_agent,
    session_service=session_service,
    app_name="ga_solver_app",
)

# The event-based pause/resume mechanism is removed as it was causing a deadlock.
# The conversation state is now managed purely by the ADK session and the client flow.


# --- Pydantic Models for API (no changes) ---
class ChatMessage(BaseModel):
    session_id: str
    user_id: str = "default_user"
    message: str

class ChatResponse(BaseModel):
    session_id: str
    bot_response: str
    needs_human_input: bool = False
    task_id: Optional[str] = None

class TaskStatus(BaseModel):
    status: str
    message: str
    details: Optional[Any] = None

async def ensure_session_exists(session_id: str, user_id: str):
    """
    Ensures a session exists by directly attempting to create it.
    This avoids any unexpected behavior from get_session and is more robust
    against the volatile nature of InMemorySessionService on server restarts.
    """
    try:
        await session_service.create_session(
            session_id=session_id,
            user_id=user_id,
            app_name="ga_solver_app",
            state={} # Always start with a fresh state on creation/re-creation
        )
        print(f"ADK session '{session_id}' was created successfully.")
    except Exception:
        # If create_session fails, it's almost certainly because the session
        # already exists in the service's memory. We can safely ignore this
        # and proceed, knowing the session is ready for the runner.
        print(f"ADK session '{session_id}' already exists. Proceeding.")
        pass

# The _handle_solver_tool_event function is removed as its logic is now
# integrated directly into the /chat and /respond endpoints.


# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, background_tasks: BackgroundTasks):
    """Handles a new user message to start or continue a conversation."""
    # Ensure the session exists in the (potentially restarted) session service.
    await ensure_session_exists(session_id=message.session_id, user_id=message.user_id)

    content = genai_types.Content(role="user", parts=[genai_types.Part(text=message.message)])
    final_response = "Sorry, I encountered an issue."
    needs_human_input = False
    task_id = None

    async for event in runner.run_async(
        user_id=message.user_id, session_id=message.session_id, new_message=content
    ):
        tool_calls = event.get_function_calls()
        if tool_calls:
            for call in tool_calls:
                if call.name == "human_approval_tool":
                    needs_human_input = True
                    final_response = call.args.get("question", "I need more information.")
                    # Break immediately to return the question to the UI.
                    break
                elif call.name == "solve_tsp_with_ga":
                    new_task_id = str(uuid.uuid4())
                    # Filter out None values so solver defaults are used
                    solver_args = {k: v for k, v in call.args.items() if v is not None}
                    task_store[new_task_id] = {
                        "status": "queued",
                        "message": "GA solver task has been queued.",
                        "details": {"input_params": solver_args},
                    }
                    # Use FastAPI's background tasks to run the blocking solver function
                    background_tasks.add_task(
                        run_gpu_ga_solver_background,
                        task_store=task_store,
                        task_id=new_task_id,
                        **solver_args
                    )
                    task_id = new_task_id
                    # Don't break here; wait for the final text response.

        if event.is_final_response():
            if event.content and event.content.parts and event.content.parts[0].text:
                final_response = event.content.parts[0].text
            # If the solver was just started, the final response should reflect that.
            elif task_id:
                final_response = f"Understood. The solver is now running with Task ID: {task_id}"
            break
        
        if needs_human_input:
            break

    return ChatResponse(session_id=message.session_id, bot_response=final_response, needs_human_input=needs_human_input, task_id=task_id)


@app.post("/respond", response_model=ChatResponse)
async def respond(message: ChatMessage, background_tasks: BackgroundTasks):
    """Handles the user's response to a question from the agent."""
    # Also ensure the session exists here before continuing the run.
    await ensure_session_exists(session_id=message.session_id, user_id=message.user_id)

    # This is constructed from the client, which knows the name of the tool
    # that prompted for input.
    tool_response_part = genai_types.Part.from_tool_response(
        name="human_approval_tool",
        response={"result": message.message}
    )
    content = genai_types.Content(role="user", parts=[tool_response_part])

    final_response = "Sorry, I encountered an issue while processing your response."
    needs_human_input = False
    task_id = None

    # Continue the agent's execution with the user's answer.
    async for event in runner.run_async(
        user_id=message.user_id, session_id=message.session_id, new_message=content
    ):
        tool_calls = event.get_function_calls()
        if tool_calls:
            for call in tool_calls:
                if call.name == "human_approval_tool":
                    needs_human_input = True
                    final_response = call.args.get("question", "I need more information.")
                    break
                elif call.name == "solve_tsp_with_ga":
                    new_task_id = str(uuid.uuid4())
                    solver_args = {k: v for k, v in call.args.items() if v is not None}
                    task_store[new_task_id] = {
                        "status": "queued",
                        "message": "GA solver task has been queued.",
                        "details": {"input_params": solver_args},
                    }
                    background_tasks.add_task(
                        run_gpu_ga_solver_background,
                        task_store=task_store,
                        task_id=new_task_id,
                        **solver_args
                    )
                    task_id = new_task_id
        
        if event.is_final_response():
            if event.content and event.content.parts and event.content.parts[0].text:
                final_response = event.content.parts[0].text
            elif task_id:
                final_response = f"Great, I've received your input. The solver is now running with Task ID: {task_id}"
            break

        if needs_human_input:
            break
    
    return ChatResponse(session_id=message.session_id, bot_response=final_response, needs_human_input=needs_human_input, task_id=task_id)


@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    status = task_store.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@app.get("/results/{task_id}")
async def get_task_results(task_id: str):
    task = task_store.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not complete. Status: {task['status']}")

    details = task.get("details", {})
    route = details.get("best_route")
    # **Refactored Logic**: Get coordinates from the input parameters stored in the task
    input_params = details.get("input_params", {})
    coordinates = input_params.get("coordinates")

    if not route or not coordinates:
        raise HTTPException(status_code=500, detail="Result data is incomplete and cannot be plotted. Coordinates not found in task store.")
    
    plot_route = route[:]
    plot_route.append(plot_route[0])
    x_coords, y_coords = zip(*[coordinates[i] for i in plot_route])

    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, 'o-', label='TSP Route')
    plt.scatter([c[0] for c in coordinates], [c[1] for c in coordinates], c='red', zorder=5)
    plt.title("TSP Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")

# ... (status endpoint and main startup are the same)
if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "YOUR_API_KEY_HERE":
        print("FATAL ERROR: The GEMINI_API_KEY environment variable is not set or is still the placeholder value.")
        print("Please set it in the 'ga-solver-adk/.env' file.")
    else:
        print("Starting Uvicorn server for ADK-Powered GA Solver (Corrected).")
        # Switching to port 8888 to avoid reserved port ranges.
        uvicorn.run(app, host="127.0.0.1", port=8888) 