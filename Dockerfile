# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 and essential tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.6 support
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install IFBench dependencies
RUN pip install --no-cache-dir -r requirement-ifbench.txt

# Install the project
RUN pip install --no-cache-dir -e .

# Install flash-attention and other GPU-specific packages
RUN pip install --no-cache-dir flash-attn==2.8.3

# Install vllm
RUN pip install --no-cache-dir vllm==0.10.1.1

# Set default command
CMD ["/bin/bash"] 