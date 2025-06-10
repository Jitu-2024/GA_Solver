# Stage 1: Build the C++/CUDA solver
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Install all necessary build dependencies directly.
# No more vcpkg.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    build-essential \
    git \
    python3-dev \
    pybind11-dev

# Copy the entire project into the build directory
WORKDIR /build
COPY . .

# Configure and build the solver with a standard CMake command.
RUN cmake -S . -B build && \
    cmake --build build

# ---

# Stage 2: Create the final Python application image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Set the PYTHONPATH environment variable
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Copy Python requirements from the builder
COPY --from=builder /build/ga-solver-simple/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code from the builder
COPY --from=builder /build/ga-solver-simple/backend/ .

# Copy the compiled solver executable from the builder stage
# The correct path is in the build directory, and the executable is named ga_solver_executable
COPY --from=builder /build/build/ga_solver_executable /app/ga_solver

# The command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 