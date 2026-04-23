# Base: GPU-enabled PyTorch
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /workspace/assignment

# Upgrade pip
RUN pip install --upgrade pip

# Core ML + VLM stack
RUN pip install \
    torch torchvision torchaudio \
    transformers \
    accelerate \
    peft \
    bitsandbytes \
    datasets \
    opencv-python \
    pillow \
    matplotlib \
    scikit-learn \
    tqdm

# nuScenes support
RUN pip install nuscenes-devkit

# Optional: jupyter (for debugging)
RUN pip install jupyter

# Default shell
CMD ["/bin/bash"]