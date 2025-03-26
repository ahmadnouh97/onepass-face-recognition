# Use the official Python 3.11 image from the Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed by DeepFace
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create the directory for DeepFace weights
RUN mkdir -p /root/.deepface/weights

# Download the facenet_weights.h5 during the build process
RUN wget https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5 -O /root/.deepface/weights/facenet_weights.h5

# Copy the requirements file into the container (if you have one)
COPY requirements.txt .

# Install the Python dependencies (if you have a requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container
COPY . .

# Command to run the Python script
CMD ["python", "app.py"]
