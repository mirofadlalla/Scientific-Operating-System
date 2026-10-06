---
title: Scientific Operating System
emoji: 🧬
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 🧬 Scientific Operating System (AI-lixir)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-enabled-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Technical Docs](https://img.shields.io/badge/Docs-Dense%20Deep%20Dive-blueviolet.svg)](README.dense.md)

> **AI-lixir**: A high-performance, multi-agent AI operating system built for Drug Discovery, Cheminformatics, and Biomedical Research with dual Text & Real-Time Voice interfaces.

📖 **Looking for comprehensive technical specs, internal architectures, schemas, and deep dives?**  
👉 Check out the [**Full Dense Technical Documentation (README.dense.md)**](README.dense.md).

---

## 🚀 Quick Highlights

- **Specialized Multi-Agent Swarm**: Intelligent routing across dedicated domain agents (Chemical, Medical, Customer Support RAG, and Conversational).
- **RAG & Scientific Knowledge Engine**: Vector indexing and retrieval supporting PubChem, PubMed, clinical guidelines, and custom uploaded research documents.
- **Multimodal Voice & Text Pipeline**: Real-time bi-directional audio WebSocket channel powered by Silero VAD, Groq Whisper STT, and fast TTS.
- **Cheminformatics Tooling**: Integrated Model Context Protocol (MCP) server for SMILES analysis, RDKit molecular descriptor generation, and PubChem retrieval.
- **Production-Ready & Observable**: Built-in dashboard (`/monitor`), OpenTelemetry metrics, Prometheus scrapers, and automated Docker healthchecks.

---

## 🏛 High-Level Architecture

```
                       ┌──────────────────────┐
                       │   Client / Frontend  │
                       │   (HTTP & WebSocket) │
                       └──────────┬───────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │         FastAPI Application          │
               │  - Auth & Rate Limiting              │
               │  - Voice Pipeline (VAD / STT / TTS)  │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │          Orchestrator Brain          │
               │        (Intent Classification)       │
               └──┬───────────────┬────────────────┬──┘
                  │               │                │
                  ▼               ▼                ▼
          ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
          │Chemical Agent│ │Medical Agent │ │  RAG Agent   │
          │ (RDKit / MCP)│ │ (PubMed/Bio) │ │(Vector Search│
          └──────────────┘ └──────────────┘ └──────────────┘
```

---

## ⚡ Quick Start

### 1. Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/)
- Alternatively, Python 3.11+ with [uv](https://github.com/astral-sh/uv) or `pip`

### 2. Clone & Configure
```bash
git clone https://github.com/mirofadlalla/Scientific-Operating-System.git
cd Scientific-Operating-System

# Copy environment template and fill in required API keys (e.g., GROQ_API_KEY)
cp .env.example .env
```

### 3. Run with Docker (Recommended)
```bash
docker compose up --build
```
The application will be accessible at:
- **API & Docs**: [http://localhost:7860/docs](http://localhost:7860/docs)
- **Monitoring Dashboard**: [http://localhost:7860/monitor](http://localhost:7860/monitor)
- **Health Check**: [http://localhost:7860/health](http://localhost:7860/health)

### 4. Local Development Setup
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

---

## 🤖 Agents & Capabilities

| Agent | Focus Area | Key Tools & Capabilities |
| :--- | :--- | :--- |
| **🧪 Chemical Agent** | Cheminformatics & Drug Discovery | SMILES parsing, PubChem queries, RDKit descriptors, molecular validation |
| **🩺 Medical Agent** | Clinical & Biomedical Science | PubMed literature lookup, disease/pathology queries, medical entity recognition |
| **📚 RAG Agent** | Document Search & Support | Vector database search, technical docs Q&A, multi-source chunk synthesis |
| **💬 Conversational Agent** | General Interaction | System routing, general dialogue, conversational context management |

---

## 📡 API Reference Overview

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/orchestrate` | `POST` | Multi-agent orchestrator streaming text endpoint |
| `/api/v1/ws/voice` | `WebSocket` | Real-time bi-directional voice channel (Silero VAD + Groq Orpheus) |
| `/api/v1/audio/transcribe` | `POST` | Whisper speech-to-text with hallucination blocklist |
| `/api/v1/audio/synthesize` | `POST` | Text-to-speech audio synthesis (WAV streaming) |
| `/api/v1/rag/ingest` | `POST` | Ingest research papers and markdown documents |
| `/api/v1/rag/status` | `GET` | Knowledge base status & embedding health check |
| `/api/v1/metrics` | `GET` | In-process telemetry, TTFT, token cost, & latency metrics |
| `/health` | `GET` | System health probe |
| `/docs` | `GET` | Interactive OpenAPI Swagger UI |

*For complete request/response schemas and examples, refer to [README.dense.md](README.dense.md#12-api-reference) or the interactive Swagger docs at `/docs`.*

---

## 📚 Complete Technical Documentation

For an in-depth breakdown of every subsystem, please explore [**`README.dense.md`**](README.dense.md):

- [System Architecture & Lifecycle](README.dense.md#2-architecture)
- [Project Layout & Architecture Layers](README.dense.md#3-project-structure)
- [Orchestrator Routing Rules & Prompt Templates](README.dense.md#6-orchestrator-brain)
- [Agent Implementations & Chemical MCP Server](README.dense.md#8-agents)
- [RAG Ingestion, Chunking Strategies & Embeddings](README.dense.md#9-rag-pipeline-rag-package)
- [Dual Memory Systems (Short-Term Deque & Long-Term Redis/JSON)](README.dense.md#10-memory-system)
- [Audio, Groq Orpheus TTS & Silero VAD v5](README.dense.md#11-audio-pipeline)
- [Detailed API Reference & WebSocket Frame Protocol](README.dense.md#12-api-reference)
- [In-Process Telemetry & Monitoring Dashboard](README.dense.md#13-monitoring-system)
- [CI/CD & Hugging Face Spaces Deployment](README.dense.md#18-deployment-guide-hf-spaces)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
