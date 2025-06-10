import asyncio
from google.adk.agents import LlmAgent, Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from config import GEMINI_MODEL

# The tool is now defined in ga_solver.py, so we import it from there.
from tools.ga_solver import create_ga_solver_tool

def create_agent() -> Agent:
    """Creates the GA solver agent."""
    return LlmAgent(
        name="ga_solver_agent",
        model=GEMINI_MODEL,  # Using a capable model
        instruction="You are a helpful assistant for running a GPU-based GA TSP solver. "
                    "Your goal is to understand the user's request, extract the parameters for the solver, "
                    "and then run the solver using the 'run_gpu_ga_tsp_solver' tool. "
                    "The tool will start a background job and return a task ID. "
                    "You MUST inform the user of this task_id so they can check the status later. "
                    "Do not wait for the result. Your job is only to start the task and report the ID.",
        tools=[create_ga_solver_tool()],
    )

def get_adk_runner() -> Runner:
    """Initializes and returns the ADK runner."""
    agent = create_agent()
    session_service = InMemorySessionService()
    
    # Define constants for identifying the interaction context
    APP_NAME = "ga_solver_app"
    
    

    # The runner is now initialized with the correct session_service.   
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    return runner

if __name__ == "__main__":
    # Example of how to use the runner.
    # Note: To test this locally, you need a Redis server and a Celery worker running.
    # 1. Start Redis: redis-server
    # 2. Start Celery worker from the `ga-solver-simple/backend` directory:
    #    celery -A celery_worker.celery_app worker --loglevel=info
    # 3. Run this script.
    async def main():
        runner = get_adk_runner()
        user_id = "test_user"
        session_id = "test_session"
        
        # Create a session
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id
        )

        # First interaction
        query1 = "Can you run the solver with 512 cities and 1000 generations?"
        print(f"User: {query1}")
        # This will now print a dict with a celery task ID
        async for event in runner.run_async(query=query1, user_id=user_id, session_id=session_id):
            if event.content:
                print(f"Agent: {event.content}")
        
        print("\nTo check the status, you would poll the /status/{task_id} endpoint.")


    asyncio.run(main()) 