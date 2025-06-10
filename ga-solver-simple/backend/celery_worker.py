from celery import Celery
from celery.result import AsyncResult
from typing import Dict, Any, Optional
import os

# Assuming this file is in ga-solver-simple/backend/
# We need to add the backend directory to the path so celery can find the `tools` module
# when run from the project root.
import sys
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# --- Celery Configuration ---
# The Redis URL is now read from an environment variable for Docker compatibility.
# It defaults to localhost for non-Docker setups.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ga_solver_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["celery_worker"] # Points to this module to find tasks
)

# Optional Celery configuration
celery_app.conf.update(
    task_track_started=True,
    result_extended=True, # To get more result metadata
)

# --- Task Definition ---
# The logic from run_gpu_ga_solver_background is now a Celery task.
# This makes it independent of the web server process.
@celery_app.task(bind=True)
def run_solver_task(self, **kwargs):
    """
    Celery task to run the GPU GA solver.
    'bind=True' gives us access to 'self' (the task instance),
    which allows us to update the task's state/metadata.
    """
    # Import the solver function here to avoid circular dependencies
    # and ensure it's loaded within the worker process.
    from tools.ga_solver import run_gpu_ga_solver

    def progress_callback(status: str, progress: Optional[float], result: Any):
        """A callback function that the solver can use to update the Celery task's state."""
        meta = {'progress': progress, 'log': result}
        self.update_state(state=status.upper(), meta=meta)
        print(f"Task {self.request.id} status updated: {status.upper()}, {meta}")

    # The `run_gpu_ga_solver` function is now a standard blocking function
    # that is executed by the Celery worker. The web app is not blocked.
    # We pass our callback to it.
    final_result = run_gpu_ga_solver(
        progress_callback=progress_callback,
        **kwargs
    )
    
    # The return value of the task is the final result.
    return final_result


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Queries Celery for the status of a task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    status_map = {
        'PENDING': 'queued',
        'STARTED': 'running',
        'SUCCESS': 'completed',
        'FAILURE': 'failed',
        'RETRY': 'running',
        'REVOKED': 'failed',
    }
    
    status = status_map.get(task_result.state, 'unknown')
    result_data = None
    progress = None

    if task_result.state == 'FAILURE':
        result_data = str(task_result.info)  # Exception info
    elif task_result.state == 'SUCCESS':
        result_data = task_result.result # Final return value
        progress = 100.0
    elif task_result.info and isinstance(task_result.info, dict):
        # For custom states like 'RUNNING'
        progress = task_result.info.get('progress')
        result_data = task_result.info.get('log')
    
    return {
        "status": status,
        "progress": progress,
        "result": result_data
    } 