---
title: Scientific Operating System
emoji: 🧬
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 🧬 AI Scientific OS - Multi-Agent Drug Discovery Hub

A production-ready, enterprise-grade multi-agent AI system for drug discovery, molecular analysis, biomedical research, and intelligent customer support. Built with FastAPI, Groq LLM, Weaviate RAG, and specialized domain agents.

**Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces)  
**Author:** Scientific Computing Lab  
**License:** MIT | **Version:** 1.0.0 | **Status:** Production Ready ✅

> ⚡ Real-time LLM routing • 🧪 Multi-agent analysis • 🧠 Persistent memory • 📡 Streaming responses • 🎤 Audio I/O

---

## 📋 Table of Contents

1. [Features](#-features) | 2. [Architecture](#-system-architecture) | 3. [Installation](#-installation--setup)  
4. [Configuration](#-configuration) | 5. [API Endpoints](#-api-endpoints) | 6. [Components](#-components-deep-dive)  
7. [Usage Examples](#-usage-examples) | 8. [Deployment](#-deployment) | 9. [Performance](#-performance-metrics) | 10. [Troubleshooting](#-troubleshooting)

---

## ✨ Features

### 🧪 Multi-Agent System
- **Chemical Agent**: Molecular analysis, ADMET predictions, drug similarity search, virtual screening
- **Medical Agent**: Biomedical pathway analysis, drug-target interactions, clinical reasoning  
- **RAG Agent**: AI-Lixir documentation retrieval and knowledge base Q&A
- **App Agent**: General system assistance and FAQ support

### 🧠 Intelligent Orchestration
- LLM-driven intent classification (Qwen3-32B / Llama-3.3-70B)
- Automatic entity extraction (compound names, SMILES, diseases)
- Dynamic routing to optimal agent
- Greeting detection skips orchestration for <100ms responses

### 🧬 Chemical Computing
- **ADMET Predictions**: Absorption, Distribution, Metabolism, Excretion, Toxicity scores
- **Virtual Screening**: Drug repurposing candidate identification
- **Similarity Search**: Chemical structure comparison via FAISS
- **Validation**: SMILES parsing and molecular fingerprinting

### 📚 Knowledge Retrieval (RAG)
- Hybrid vector + BM25 search via Weaviate
- Groq embeddings (llama-text-embed-v2)
- Recursive document chunking
- Context-aware synthesis

### 💾 Dual-Layer Memory
- **Short-Term**: Session in-memory cache (thread-safe, 50 msg max)
- **Long-Term**: Redis-backed persistent storage (optional)
- Automatic conversation history tracking

### 🎤 Audio Integration
- Speech-to-text: Groq Whisper (99+ languages)
- Text-to-speech: Kokoro TTS + Browser API + OpenAI TTS
- WebSocket streaming for real-time interaction
- Support for WAV, MP3, FLAC, OGG formats

### 🌐 Multilingual Support
- Arabic, English, code-mixed queries
- RTL text handling
- Language-agnostic routing

### 🐳 Enterprise Ready
- Docker + Docker Compose
- Hugging Face Spaces compatible
- Health checks & readiness probes
- Background task processing (RQ)
- Redis & Weaviate integration

---

## 🏗️ System Architecture

### Processing Flow

```
Input (Text/Audio) → Greeting? → [YES] Direct Response → Memory Update → Output
                         ↓
                        [NO]
                         ↓
                    Orchestrator (Intent + Entity Extraction)
                         ↓
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
    Chemical         Medical            RAG        App
    Agent            Agent            Agent       Agent
         ↓               ↓               ↓
         └───────────────┼───────────────┘
                         ↓
              Synthesis Layer (Combine outputs)
                         ↓
          Streaming Response Engine (SSE/WebSocket)
                         ↓
              Memory Update (Short + Long-term)
```

### Component Stack

```
FastAPI Application (Port 7860)
├── Request Handlers
│   ├── POST /summarize-text (text queries)
│   ├── POST /stream-response (streaming)
│   ├── WebSocket /ws (bidirectional)
│   ├── POST /process-audio (voice input)
│   └── GET /health (status)
│
├── Orchestration Layer
│   └── OrchestratorBrain
│       ├── LLM Intent Classification
│       ├── Entity Extraction
│       └── Dynamic Router
│
├── Agent Layer (Parallel Execution)
│   ├── ChemicalAgent → ADMET/Screening services
│   ├── MedicalAgent → Groq LLM (biomedical)
│   ├── CustomerSupportRAGAgent → Weaviate
│   └── AppAgent → FAQ fallback
│
├── Storage Layer
│   ├── ShortTermMemory (in-memory deque)
│   └── LongTermMemory (Redis optional)
│
└── Response Engine
    ├── Output Synthesis
    └── SSE/WebSocket Streaming
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (recommended)
- Groq API Key: [console.groq.com](https://console.groq.com)
- 4GB+ RAM, 2GB+ Disk

### Option 1: Local Development

```bash
git clone <repo-url> && cd "Scientific Operating System"
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip && pip install -r requirements.txt
cp .env.example .env  # Edit: add GROQ_API_KEY
python main.py  # Runs on http://localhost:7860
```

### Option 2: Docker Compose (Full Stack)

```bash
git clone <repo-url> && cd "Scientific Operating System"
cp .env.example .env  # Add GROQ_API_KEY
docker-compose up --build  # Starts: FastAPI (7860) + Redis + Weaviate
# View logs: docker-compose logs -f scientific-os
# Stop: docker-compose down -v
```

### Option 3: Hugging Face Spaces

1. Create new Space (Docker SDK)
2. Upload repository files
3. Set `GROQ_API_KEY` in Secrets
4. HF auto-builds and deploys on port 7860

---

## ⚙️ Configuration

### Required Environment Variables

```bash
GROQ_API_KEY=gsk_...  # Get from console.groq.com
```

### Optional Variables (with defaults)

```bash
# LLM Models
GROQ_BASE_URL=https://api.groq.com/openai/v1
ORCHESTRATOR_MODEL=llama-3.3-70b-versatile
QWEN_MODEL=qwen/qwen3-32b

# Embeddings
EMBEDDING_PROVIDER=huggingface  # or: openai
EMBEDDING_MODEL=BAAI/bge-m3
OPENAI_API_KEY=  # If using OpenAI embeddings/TTS

# External Services
ADMET_AI_URL=https://shdwRow-ailixir-admet.hf.space
CHEMICAL_AI_URL=https://RottenShadow-ailixir-chemical-rag.hf.space
DRUG_REPURPOSING_URL=https://RottenShadow-ailixir-drug-repurposing.hf.space
GENERATION_SERVICE_URL=https://shdwRow-ailixir-generation.hf.space

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Weaviate (optional)
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
```

---

## 🔌 API Endpoints

### 1. Text Query: `POST /summarize-text`

```bash
curl -X POST http://localhost:7860/summarize-text \
  -H "Content-Type: application/json" \
  -d '{
    "text_input": "What'\''s ADMET of aspirin?",
    "session_id": "user-123",
    "user_id": "user-456"
  }'
```

**Response (Streaming SSE):**
```
data: {"type": "start_streaming", "agent": "CHEMICAL_AGENT"}
data: {"type": "token", "content": "ADMET analysis"}
data: {"type": "complete"}
```

### 2. Streaming Response: `POST /stream-response`

Query with streaming output (JSON/SSE)

### 3. Real-Time WebSocket: `WebSocket /ws`

Bidirectional communication for live updates

### 4. Audio Processing: `POST /process-audio`

```bash
curl -X POST http://localhost:7860/process-audio \
  -F "file=@audio.wav" \
  -F "session_id=session-123"
```

### 5. Health Check: `GET /health`

Returns system status and service availability

### 6. API Documentation: `GET /docs`

Interactive Swagger UI with all endpoints

---

## 🔍 Components Deep Dive

### 1. Orchestrator Brain (`app/orchestrator/brain.py`)

**Intent Classification:**

| Intent | Agent | Use Case |
|--------|-------|----------|
| CHEMICAL_SIMILARITY | Chemical | Molecular structure search |
| ADMET_ANALYSIS | Chemical | Drug property prediction |
| DRUG_REPURPOSING | Chemical+Medical | Disease screening |
| BIOMEDICAL_MECHANISM | Medical | Pathway analysis |
| APP_SUPPORT_RAG | RAG | Documentation Q&A |
| APP_HELP | App | General FAQ |

**Routing:** User query → LLM classification → Entity extraction → Agent selection → Execution

---

### 2. Chemical Agent (`app/agents/chemical/agent.py`)

**ADMET Prediction**
- Input: SMILES string
- Service: `{ADMET_AI_URL}/predict_batch`
- Output: Absorption, Distribution, Metabolism, Excretion, Toxicity (0-1 scores)

**Virtual Screening**
- Input: Disease name
- Service: `{DRUG_REPURPOSING_URL}/api/v1/screen`
- Output: Top drug candidates with binding scores

**Similarity Search**
- Input: SMILES
- Service: `{CHEMICAL_AI_URL}/search/{mode}`
- Modes: full-rag (with reasoning), retrieval-only (fast)

---

### 3. Medical Agent (`app/agents/medical/agent.py`)

**Biomedical Reasoning**
- Drug-target interactions
- Mechanism of action analysis
- Disease therapeutic rationale
- Pharmacological evaluation

**LLM Backend:** Groq (llama-3.3-70b, temperature=0.0 for precision)

---

### 4. Customer Support RAG (`app/agents/customer_support/agent.py`)

**RAG Pipeline:**
1. Query embedding (Groq llama-text-embed-v2)
2. Weaviate hybrid search (50% BM25, 50% vector)
3. Top-4 chunk retrieval
4. LLM synthesis (llama-3.3-70b)
5. Grounded response

**Knowledge Base:** ADMET service docs, API specs, system guides

---

### 5. Memory System

**Short-Term** (`app/memory/short_term.py`)
- Thread-safe in-memory deque
- Per-session history (max 50 messages)
- Instant availability

**Long-Term** (`app/memory/long_term.py`)
- Optional Redis backend
- Cross-session persistence
- Graceful fallback if unavailable

---

## 💡 Usage Examples

### Example 1: ADMET Analysis

**Query:** "What's the ADMET of aspirin (CC(=O)Oc1ccccc1C(=O)O)?"  
**Flow:** Orchestrator → Detect ADMET_ANALYSIS → Chemical Agent → ADMET service  
**Output:**
```
[ADMET Analysis for CC(=O)Oc1ccccc1C(=O)O]:
• Absorption: 0.8743
• Distribution: 0.9201
• Metabolism: 0.7821
• Excretion: 0.8145
• Toxicity: 0.1456
```

### Example 2: Virtual Screening

**Query:** "Screen compounds for Alzheimer's treatment"  
**Flow:** Orchestrator → Detect DRUG_REPURPOSING → Chemical + Medical agents  
**Output:** Top drug candidates with therapeutic rationale

### Example 3: Documentation

**Query:** "How to use ADMET API?"  
**Flow:** Orchestrator → Detect APP_SUPPORT_RAG → RAG Agent → Weaviate search  
**Output:** API documentation with examples

### Example 4: Greeting (Optimized)

**Query:** "مرحبا، كيفك؟" (Arabic greeting)  
**Flow:** Greeting detection → Direct response (skip orchestrator)  
**Latency:** ~50-100ms (vs. 300-500ms with full routing)

---

## 🚀 Deployment

### Local with Hot Reload

```bash
python main.py  # Runs on :7860
```

### Docker Single Container

```bash
docker build -t scientific-os:latest .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key scientific-os:latest
```

### Docker Compose Full Stack

```bash
docker-compose up --build -d
# Starts: FastAPI (7860), Redis (6379), Weaviate (8080)
docker-compose down -v  # Stop all + remove volumes
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Greeting → Response | <100ms |
| First-token latency | <300ms |
| ADMET prediction | 1-2s |
| Virtual screening | 5-10s |
| RAG query | 1-3s |
| Full multi-agent synthesis | 3-5s |
| Memory per session | 50-100MB |

**Optimization:**
- Greeting detection skips orchestration (-200ms)
- Parallel agent execution
- Response streaming (perceived latency reduction)

---

## 🐛 Troubleshooting

### GROQ_API_KEY not set
```bash
export GROQ_API_KEY="gsk_..."
# or add to .env file
```

### External services unreachable
```bash
# Check URLs
curl -I https://shdwRow-ailixir-admet.hf.space
# Check firewall settings
```

### Weaviate connection refused
```bash
# Ensure Weaviate is running
docker-compose up -d weaviate
# Check connectivity
curl http://localhost:8080/v1/.well-known/ready
```

### Redis timeout
```bash
# Option 1: Start Redis
docker run -d -p 6379:6379 redis:7

# Option 2: Disable (uses in-memory fallback)
export REDIS_HOST=""
```

### Docker build fails
```bash
# Clean build cache
docker builder prune

# Rebuild with verbose
docker build -t scientific-os:latest --progress=plain .
```

---

## 🔐 Security Best Practices

✅ Store API keys in environment variables (never commit)  
✅ Use .env file (add to .gitignore)  
✅ HTTPS for all external calls  
✅ Input validation for SMILES/queries  
✅ Rate limiting per session  
✅ SSL/TLS in production (reverse proxy)

---

## 📚 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | FastAPI + Uvicorn |
| **LLM** | Groq (Llama-3.3-70B, Qwen-3.2-32B) |
| **Embeddings** | Hugging Face BAAI/bge-m3, Groq |
| **Vector DB** | Weaviate (hybrid BM25+vector) |
| **RAG** | LlamaIndex with recursive chunking |
| **Voice** | Groq Whisper STT, Kokoro TTS |
| **Storage** | Redis, JSON file, in-memory |
| **Containerization** | Docker, Docker Compose |
| **Testing** | pytest, pytest-asyncio |

---

## 🗺️ Roadmap

**Near-Term:** Chromadb integration, RDKit visualization, metrics dashboard  
**Mid-Term:** Advanced voice features, real-time collaboration, semantic cleanup  
**Long-Term:** Federated learning, custom fine-tuning, pharma DB integration

---

## 🤝 Contributing

Fork → Feature branch → Commit → Push → Pull Request

---

## 📞 Support

**Issues:** GitHub Issues | **Discussions:** HF Spaces | **Docs:** README_FULL.md

---

## 📄 License

MIT License - See LICENSE file

---

**Last Updated:** June 2026 | **Version:** 1.0.0 | **Status:** Production Ready ✅

🧬 AI Scientific OS © 2026 | Powered by Groq | Deployed on Hugging Face Spaces
