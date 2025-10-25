# Base Docker Image of verl, with CUDA/Torch/FlashAttn/Apex/TransformerEngine, without other frameworks
# Target: verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.8.0-fi0.2.6
# Start from the NVIDIA official image (ubuntu-22.04 + cuda-12.6 + python-3.10)
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html
FROM hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0
# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# # Install project dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# Install IFBench dependencies
RUN pip install --no-cache-dir -r requirement-ifbench.txt

# Install the project
RUN pip install --no-cache-dir -e .

# # Install flash-attention and other GPU-specific packages
# RUN pip install --no-cache-dir flash-attn==2.8.3

# Install vllm
RUN pip install --no-cache-dir vllm==0.10.1.1

RUN pip install --no-cache-dir wandb

RUN pip install --no-cache-dir huggingface-hub

RUN pip uninstall -y flash-attn

RUN pip install flash-attn --no-build-isolation

RUN pip install --no-cache-dir gpustat

RUN pip install --no-cache-dir nvitop

RUN git clone https://github.com/mianzhang/jsonparse

RUN cd jsonparse && pip install -e .

# Set default command
CMD ["/bin/bash"] 