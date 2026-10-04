# syntax=docker/dockerfile:1.7
# ═════════════════════════════════════════════════════════════════════════════
#  AI-lixir Scientific OS — backend image (FastAPI + MCP chemical server)
#
#  Stages
#    builder      venv with CPU-only torch + the pinned requirements.txt
#    model-cache  (optional) pre-downloads the embedding model into /opt/hf
#    runtime      slim, non-root image: venv + model cache + app code only
#
#  Build:   docker build -t scientific-os-backend .
#  Preload:  docker build --build-arg PRELOAD_MODEL=1 -t scientific-os-backend .
#            (Only needed for EMBEDDING_PROVIDER=huggingface. Adds ~560 MB but
#             avoids the 2-3 min cold-start download on first container run.)
#            For EMBEDDING_PROVIDER=jina or openai, always leave PRELOAD_MODEL=0.
# ═════════════════════════════════════════════════════════════════════════════
ARG PYTHON_VERSION=3.11


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder: resolve nothing, just install exactly what is pinned
# ─────────────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ARG UV_VERSION=0.11.7
# torch is NOT in requirements.txt on purpose: PyPI's default Linux wheel drags in
# ~3 GB of CUDA libraries that a CPU-only Groq-backed service never uses.
ARG TORCH_SPEC="torch>=2.4,<3"
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# No compiler / apt packages needed: every pinned dependency ships a wheel for
# linux x86_64 and aarch64, and --only-binary below makes that a hard guarantee.
RUN pip install --no-cache-dir uv==${UV_VERSION} && uv venv /opt/venv

# Layer A — CPU torch. Changes rarely, so it stays cached across code/dep edits.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url "${TORCH_INDEX}" "${TORCH_SPEC}"

# Layer B — everything else, exactly as pinned (no resolution happens here).
#   --no-deps       the lock is complete; nothing new may sneak in
#   --only-binary   never compile from source
#   uv pip check    build FAILS if any package conflicts (incl. against torch)
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps --only-binary :all: -r /tmp/requirements.txt \
 && uv pip check

# Guard: the image must be CPU-only. Fails the build if torch is a CUDA build or
# if any NVIDIA / triton wheel slipped into the venv.
RUN python - <<'PY'
import importlib.metadata as md, sys, torch
bad = sorted(d.metadata["Name"] for d in md.distributions()
             if d.metadata["Name"].lower().startswith(("nvidia-", "cuda-", "triton")))
if torch.version.cuda is not None or bad:
    sys.exit(f"Not CPU-only: torch.version.cuda={torch.version.cuda!r}, GPU packages={bad}")
print("CPU-only OK:", torch.__version__)
PY

# Bake llama-index's import-time downloads (NLTK punkt_tab + stopwords,
# tiktoken BPE) into the venv so the running container never needs the network.
#
# GlobalsHelper.wait_for_nltk_check() is lazy — a bare `import llama_index.core`
# instantiates GlobalsHelper but never calls wait_for_nltk_check(), so the
# NLTK corpora are NOT written to disk.  Accessing .stopwords calls
# wait_for_nltk_check(), which downloads both punkt_tab and stopwords.
RUN python - <<'PY'
from llama_index.core.utils import globals_helper
# .stopwords triggers wait_for_nltk_check() → downloads punkt_tab + stopwords
# into _static/nltk_cache inside the venv (confirmed by CI build log).
_ = globals_helper.stopwords
print("NLTK punkt_tab + stopwords baked in.")
PY


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — model-cache: embedding model baked in (skippable)
# ─────────────────────────────────────────────────────────────────────────────
FROM builder AS model-cache

# PRELOAD_MODEL=0 (default) → image contains NO embedding model weights.
# Set PRELOAD_MODEL=1 only when EMBEDDING_PROVIDER=huggingface and you want to
# bake the model into the image to avoid the cold-start download.
ARG PRELOAD_MODEL=0
# Keep in sync with EMBEDDING_MODEL in .env / app/config.py.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
ENV HF_HOME=/opt/hf

RUN mkdir -p "${HF_HOME}" \
 && if [ "${PRELOAD_MODEL}" = "1" ]; then \
      python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBEDDING_MODEL'])"; \
    fi


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

# uid 1000 matches what Hugging Face Spaces runs containers as.
ARG APP_UID=1000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false \
    PORT=7860

# tini = proper PID 1: reaps zombies (the app spawns an RQ worker subprocess and
# the entrypoint runs the MCP server) and forwards SIGTERM for clean shutdown.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid ${APP_UID} --create-home --shell /usr/sbin/nologin app \
 && mkdir -p /data /code \
 && chown app:app /data /code

COPY --from=builder /opt/venv /opt/venv
COPY --from=model-cache --chown=${APP_UID}:${APP_UID} /opt/hf /opt/hf

# The venv is copied from builder where everything ran as root.  LlamaIndex
# writes NLTK data (punkt_tab, stopwords) and tiktoken BPE cache into
#   /opt/venv/lib/python*/site-packages/llama_index/core/_static/
# at runtime if the app user can't write there.  Hand ownership of that
# specific subtree to the app user so both reads and any future writes succeed.
# We use a glob so this keeps working if the Python minor version ever changes.
RUN chown -R ${APP_UID}:${APP_UID} \
      /opt/venv/lib/python*/site-packages/llama_index/core/_static


WORKDIR /code
COPY app ./app
COPY docker/entrypoint.sh  /usr/local/bin/entrypoint.sh
COPY docker/healthcheck.py /usr/local/bin/healthcheck.py

# Paths the app writes to at runtime (see .dockerignore: no local state is baked in):
#   /data                                    persistent RAG index (HF Spaces convention)
#   app/memory                               long_term_store.json fallback
#   …/rag/storage            RAG index fallback when /data is unavailable
RUN chmod +x /usr/local/bin/entrypoint.sh \
 && python -m compileall -q app \
 && mkdir -p app/agents/customer_support/rag/storage \
 && chown -R app:app app/memory app/agents/customer_support/rag/storage

USER app
EXPOSE 7860

# /health bypasses ReadinessMiddleware, so it is green as soon as the API is up
# (even while the RAG engine is still warming). start-period covers the slow
# llama-index / torch imports on first boot.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "/usr/local/bin/healthcheck.py"]

ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/entrypoint.sh"]

# Single worker on purpose: the app keeps in-memory state, spawns its own RQ
# worker in the lifespan hook, and each worker would load the embedding model.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
