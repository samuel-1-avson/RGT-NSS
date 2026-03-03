FROM nvidia/cuda:12.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy requirements
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Environment variables for GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
