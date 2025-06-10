from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
import sys
import os
import uvicorn
from typing import List, Dict, Any, Tuple, Optional
import uuid
from contextlib import asynccontextmanager
import json
from google.adk.sessions import InMemorySessionService
# Add the parent directory of 'backend' to sys.path to allow importing ga_solver_chatbot
# This assumes 'backend' is a subdirectory of 'ga-solver-simple'
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir) # No longer needed for tools.ga_solver


# Now we can import the background solver function
from adk_agent import get_adk_runner
from google.adk.runners import Runner
# Per the tutorial, the correct import is from 'google.genai'.
from google.genai import types as genai_types
from celery_worker import get_task_status

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the ADK runner
    runners["ga_solver_runner"] = get_adk_runner()
    print("ADK Runner initialized.")
    yield
    # Clean up resources if needed
    runners.clear()
    print("ADK Runner shut down.")

app = FastAPI(
    title="GA Solver Chatbot API",
    description="API for interacting with the TSP Genetic Algorithm Solver Chatbot.",
    version="0.4.0", # Version bump for Celery integration
    lifespan=lifespan
)

# Using a dictionary to store the runner for simplicity.
runners: Dict[str, Runner] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    task_id: str | None = None

class StatusResponse(BaseModel):
    status: str
    progress: float | None = None
    result: Any | None = None
    

# async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str):
#     print(f"Calling agent with query: {query}")
    
#     content = genai_types.Content(
#         parts=[genai_types.Part(text=query)], role="user"
#     )

#     async for event in runner.run_async(
#         new_message=content, user_id=user_id, session_id=session_id
#     ):
        

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    user_id = "user_1" # In a real app, you'd get this from auth
    runner = runners["ga_solver_runner"]

    # The correct pattern is to unconditionally call create_session on the
    # runner's shared session_service. This is an idempotent "get or create" operation.
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )

    agent_reply = ""
    task_id = None
    
    try:
        # Per the tutorial, construct a Content object with role and parts.
        user_input = genai_types.Content(
            parts=[genai_types.Part(text=request.message)], role="user"
        )
        async for event in runner.run_async(
            new_message=user_input, user_id=user_id, session_id=session_id
        ):
            if event.type == "llm_response" and event.content:
                agent_reply += event.content.parts[0].text if event.content.parts else ""
            if event.type == "tool_output":
                # The event.content is the dictionary returned by our tool
                if event.content and event.content.get("tool_name") == "run_gpu_ga_tsp_solver":
                    # The output from our tool is a dictionary string
                    outputs = event.content.get("outputs", [])
                    if outputs:
                        output_data = json.loads(outputs[0].get("text", "{}"))
                        task_id = output_data.get("task_id")

        if not agent_reply:
            agent_reply = "The agent did not produce a conversational reply, but the task may have been started."

        return ChatResponse(reply=agent_reply, session_id=session_id, task_id=task_id)

    except Exception as e:
        # Re-adding diagnostic traceback printing for one final verification.
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing agent request: {e}")

@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """
    Poll this endpoint with a task_id to get the status of a running solver job.
    The status is retrieved from the Celery result backend.
    """
    status = get_task_status(task_id)
    if status.get("status") == "unknown":
        raise HTTPException(status_code=404, detail="Task not found")
    return StatusResponse(**status)

# To run this app stack:
# 1. Start Redis:
#    (If using Docker): docker run -d -p 6379:6379 redis
#    (Or natively): redis-server
#
# 2. Start the Celery Worker (from this `backend` directory):
#    celery -A celery_worker.celery_app worker --loglevel=info
#
# 3. Start the FastAPI Server (from this `backend` directory):
#    uvicorn main:app --reload
#
# 4. You can then use the test script in `tests/test_api.py`

if __name__ == "__main__":
    # This block allows running the app directly for simple testing,
    # but the full stack (Redis, Celery worker) must be running.
    print("Starting Uvicorn server. Ensure Redis and Celery workers are running.")
    print("Navigate to http://127.0.0.1:8000/docs for API documentation.")
    uvicorn.run(app, host="0.0.0.0", port=8000) 