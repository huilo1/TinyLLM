FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY configs ./configs
COPY deploy ./deploy
COPY src ./src
COPY artifacts/qwen25_1_5b_instruct_qlora_v1 ./artifacts/qwen25_1_5b_instruct_qlora_v1
COPY artifacts/tiny_rfn/tokenizer ./artifacts/tiny_rfn/tokenizer
COPY artifacts/tiny_rfn_v2/tokenizer ./artifacts/tiny_rfn_v2/tokenizer

# Start from the official PyTorch runtime image, then add only project-specific
# dependencies to keep our own layers smaller and faster to publish.
RUN uv pip install --system \
        "accelerate>=1.2.0" \
        "bitsandbytes>=0.48.0" \
        "datasets>=3.3.0" \
        "gradio>=5.23.0" \
        "matplotlib>=3.10.0" \
        "numpy>=2.1.0" \
        "peft>=0.18.0" \
        "tokenizers>=0.20.0" \
        "transformers>=4.57.0" \
        "trl>=0.23.0" \
        "tqdm>=4.67.0" \
    && uv pip install --system --no-deps -e /app

CMD ["bash"]
