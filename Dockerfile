# Base: GPU-enabled PyTorch
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

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
# RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

RUN pip install transformers==4.44.2 tokenizers==0.19.1 accelerate==0.30.1

RUN pip install bitsandbytes==0.43.1 sentencepiece==0.2.1 protobuf

# nuScenes support
RUN pip install nuscenes-devkit rouge-score bert-score

# Optional: jupyter (for debugging)
RUN pip install jupyter

# Default shell
CMD ["/bin/bash"]