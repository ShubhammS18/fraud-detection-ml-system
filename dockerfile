# Step 1: Use a lightweight Python base image
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies (needed for LightGBM/Pandas)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy only the requirements first (optimizes Docker caching)
COPY requirements.txt .

# Step 5: Install Python dependencies with increased timeout
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Step 6: Copy the project code and artifacts
COPY src/ ./src/
COPY artifacts/ ./artifacts/
# Create an empty logs folder inside the container
RUN mkdir -p logs

# Step 7: Expose the port FastAPI runs on
EXPOSE 8000

# Step 8: Run the application
# We use 0.0.0.0 to allow external access to the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]