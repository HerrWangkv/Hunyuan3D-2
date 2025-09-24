# Dockerfile for MagicDrive-V2
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3 -m pip install --upgrade pip

# Create working directory
WORKDIR /workspace/

# Install torch, torchvision, packaging and apex in one command to ensure torch is available for apex build
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install git+https://github.com/nerfstudio-project/gsplat.git

# Copy source code just before installing requirements
COPY . /workspace/
RUN pip install -r requirements.txt && pip install -e .
RUN pip install --upgrade matplotlib
RUN cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install && cd ../../../ && cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install

# Install squashfuse and ffmpeg
RUN apt-get update && apt-get install -y squashfuse fuse ffmpeg libopenmpi-dev openmpi-bin && ln -sf /usr/bin/python3 /usr/bin/python
RUN apt-get install -y \
    libglu1-mesa \
    freeglut3-dev \
    mesa-utils \
    libx11-dev \
    libxrender-dev \
    libxext-dev

# Set entrypoint
CMD ["/bin/bash"]
