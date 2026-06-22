---
title: Scientific Operating System
emoji: 🧬
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# AI Scientific OS - Multi-Agent Drug Discovery Hub

This is an advanced Multi-Agent Scientific Operating System specializing in automated virtual drug screening, ADMET property prediction, and deep biomedical mechanism analysis.

## 🚀 Deployment Architecture
- **Backend Core**: FastAPI (Asynchronous Uvicorn Router)
- **Streaming Pipeline**: Native AsyncOpenAI chunk streaming token delivery
- **Expert Networks**: Integrated Chemical Agent & Medical Agent (OpenBioLLM)
- **Containerization**: Optimized Docker container running on exposed port 7860

## 🛠️ Environment Variables Configuration
Ensure the following Secrets are set within your Hugging Face Space settings console:
- `GROQ_API_KEY`
- `GROQ_BASE_URL`
- `ORCHESTRATOR_MODEL`
- `ADMET_AI_URL`
- `DRUG_REPURPOSING_URL`
- `CHEMICAL_AI_URL`

# 🧠 SciOS: AI Scientific Operating System
### Drug Discovery + Medical RAG + Contextual Multi-Agent System

SciOS is an advanced, production-ready AI Operating System designed to bridge the gap between biomedical reasoning and chemical computation. Powered by **Qwen3** as the central orchestrator, the system routes complex scientific queries to specialized expert agents while maintaining a continuous, multi-layered memory system.

---

## ⚡ Key Features

* **🧠 Central Orchestrator (Qwen3):** Real-time intent classification, dynamic tool selection, and advanced response synthesis with sub-300ms first-token latency.
* **🧪 Chemical Expert Agent:** Molecular similarity search utilizing **FAISS (IVF Indexing)** and ADMET property prediction via **5 Graph Neural Networks (MPNNs)** over a 10M+ compound database space.
* **🧬 Medical Expert Agent:** Clinical-grade reasoning built on **OpenBioLLM** combined with a specialized Medical RAG framework for pharmacology and biomedical knowledge retrieval.
* **🧠 Tri-Tier Memory System:** Continuous context management utilizing Short-term context, Long-term user profiling, and past-query Vector Memory (ChromaDB).
* **🎙️ Voice-to-Voice Pipeline:** Seamless interaction via **Whisper Large V3 Turbo** (STT) and **Kokoro / XTTS-v2** (TTS) for hands-free laboratory operations.

---

## 📂 Project Structure


```text
ai-scientific-os/
│
├── .env                    # Environment keys & private settings
├── requirements.txt         # Required libraries (FastAPI, FAISS, OpenAI, etc.)
├── README.md               # Documentation & setup instructions
│
├── app/                    # Primary application codebase
│   ├── main.py             # FastAPI entrypoint
│   ├── config.py           # Configuration loading via pydantic-settings
│   │
│   ├── orchestrator/       # Central orchestrator & system prompts
│   │   ├── brain.py        # Intent classification & routing logic (Qwen3-32b)
│   │   └── prompts.py      # System prompts repository
│   │
│   ├── memory/             # Memory subsystem (Short-term Redis & Long-term Chroma)
│   │   ├── short_term.py   # Temporary session manager
│   │   └── long_term.py    # Chroma Vector DB wrapper
│   │
│   ├── agents/             # Expert agent definitions
│   │   ├── chemical/       # Chemical Agent (FAISS & MPNNs models)
│   │   ├── medical/        # Medical Agent (OpenBioLLM & medical RAG)
│   │   └── app_agent/      # Core Application FAQ & Support Agent
│   │
│   ├── tools/              # Core external integrations & tools
│   │   ├── db_queries.py   # Drug DB database integration
│   │   └── app_apis.py     # External backend API integrations
│   │
│   └── utils/              # Helper utilities
│       ├── audio.py        # Speech-to-Text (Whisper) & Text-to-Speech (Kokoro)
│       └── caching.py      # LRU Cache utilities
│
└── data/                   # Git-ignored local data resources
    ├── faiss_indexes/      # Saved FAISS index binaries
    ├── medical_docs/       # Document source files for Medical RAG
    └── checkpoints/        # Local MPNN model weights
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Copy or rename `.env` and fill out your API credentials.

3. **Start the API Server**:
   ```bash
   python -m uvicorn app.main:app --reload
   ```
