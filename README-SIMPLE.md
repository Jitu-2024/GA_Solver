# GA Solver Simple

This project now runs entirely within Docker.

## Prerequisites
- Docker Desktop
- An NVIDIA GPU

## How to Run
1. Open a terminal in the project root.
2. Run the command: `docker-compose up --build`

This will build the solver, start all services, and grant the solver GPU access.
The API will be available at `http://127.0.0.1:8000`.



