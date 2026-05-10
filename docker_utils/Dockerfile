# Polaris / Planning Wrapper for ManiSkill3
# GPU (NVIDIA) + GUI (X11) support, Python 3.11, Pinocchio (apt + pip), ManiSkill3, PyTorch CUDA 12.1
# No conda: system Python + pip only.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
# Require full GPU + Vulkan/OpenGL from nvidia-container-toolkit (must use --gpus all when running)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# System deps: Python 3.11, Git, OpenGL/Mesa, Vulkan, X11, GLib (for OpenCV/cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libegl1 \
    libgles2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglvnd0 \
    libglvnd-dev \
    libvulkan1 \
    libxext6 \
    libxrender1 \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    vulkan-tools \
    x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default and create venv for POLARIS (no conda)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy POLARIS into image for install (build-time); runtime will mount host over /workspace
COPY docker_ws/POLARIS /opt/POLARIS

# Upgrade pip and install Python deps. Robotics Pinocchio is PyPI package "pin" (not "pinocchio").
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "numpy>=1.22.0,<2.0.0" \
    && pip install --no-cache-dir pin \
    && pip install --no-cache-dir -r /opt/POLARIS/requirements.txt \
    && pip install --no-cache-dir git+https://github.com/haosulab/ManiSkill.git \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -e /opt/POLARIS

# Vulkan: NVIDIA uses libGLX_nvidia.so.0 as Vulkan ICD. Host libs mounted at /host-lib and /host-lib-usr by run.sh.
RUN mkdir -p /usr/share/vulkan/icd.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"/host-lib/libGLX_nvidia.so.0"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"/host-lib-usr/libGLX_nvidia.so.0"}}' > /usr/share/vulkan/icd.d/nvidia_icd_usr.json

# Default working directory (overridden by mount to host workspace)
WORKDIR /workspace

# Keep container running so run.sh/terminal.sh can attach/exec
CMD ["sleep", "infinity"]
