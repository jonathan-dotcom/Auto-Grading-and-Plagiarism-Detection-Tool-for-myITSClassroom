FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash coderunner
USER coderunner

# Set working directory
WORKDIR /app

# Run command will be defined when the container is executed
CMD ["python", "-c", "print('Code execution container is ready')"]