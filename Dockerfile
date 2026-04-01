FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY configs ./configs
COPY deploy ./deploy
COPY src ./src
COPY artifacts/tiny_rfn/tokenizer ./artifacts/tiny_rfn/tokenizer
COPY artifacts/tiny_rfn_v2/tokenizer ./artifacts/tiny_rfn_v2/tokenizer

# Install a CUDA 12.4-compatible PyTorch build explicitly to avoid hosts
# pulling a newer cu130 wheel that may not match Vast.ai driver versions.
RUN uv venv /opt/venv --python 3.11 \
    && uv pip install --python /opt/venv/bin/python \
        "datasets>=3.3.0" \
        "gradio>=5.23.0" \
        "numpy>=2.1.0" \
        "tokenizers>=0.20.0" \
        "tqdm>=4.67.0" \
    && uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu124 \
        "torch>=2.5.0,<2.11.0" \
    && uv pip install --python /opt/venv/bin/python --no-deps -e /app

CMD ["bash"]
