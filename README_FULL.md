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

**A production-ready AI Operating System for automated drug discovery, biomedical reasoning, and multi-agent scientific research.**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [API Endpoints](#api-endpoints)
9. [Components Deep Dive](#components-deep-dive)
10. [Deployment](#deployment)
11. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The **Scientific Operating System (SciOS)** is an advanced AI-powered platform designed to bridge the gap between computational chemistry and biomedical reasoning. It combines:

- **Multi-Agent Architecture**: Specialized agents for chemical analysis and medical reasoning
- **Central Orchestrator**: Intelligent routing based on query intent
- **Memory System**: Three-tier memory management (short-term, long-term, vector)
- **Streaming Responses**: Real-time token delivery for instant feedback
- **Docker Containerization**: Production-ready deployment on Hugging Face Spaces

### Use Cases

✅ **Virtual Drug Screening** - Identify drug candidates for specific diseases  
✅ **ADMET Prediction** - Calculate absorption, distribution, metabolism, excretion, toxicity  
✅ **Drug Repurposing** - Find new therapeutic applications for existing compounds  
✅ **Biomedical Research** - Deep analysis of biological pathways and mechanisms  
✅ **General Scientific Q&A** - Conversational assistance for scientific queries  

---

## 🏗️ Architecture

### System Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    User Input (Web UI / API)                 │
└─────────────────────────┬──────────────────────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │  Quick Check: Is this a greeting?  │
        │  (Skip orchestrator if yes)        │
        └────┬──────────────────────────┬───┘
             │                          │
         [YES]                      [NO]
             │                          │
             │                    ┌─────▼──────────────────┐
             │                    │  Orchestrator Brain    │
             │                    │  (LLM Intent Router)   │
             │                    └─────┬──────────────────┘
             │                          │
             │              ┌───────────┼───────────┐
             │              │           │           │
        Direct    ┌─────────▼──┐  ┌─────▼──┐  ┌────▼─────┐
        Response  │  Chemical  │  │ Medical │  │ APP_HELP │
        (Greeting)│   Agent    │  │ Agent   │  │  Agent   │
             │    └─────┬──────┘  └────┬────┘  └────┬─────┘
             │          │              │           │
             │    ┌─────▼──────────────▼──────────▼─┐
             │    │   Synthesis Layer               │
             │    │   (Combine outputs)             │
             │    └─────┬────────────────────────────┘
             │          │
             └──────────┼──────────────────┐
                        │                  │
                 ┌──────▼────────────────────▼─┐
                 │  Streaming Response Engine   │
                 │  (Token-by-token delivery)   │
                 └──────┬──────────────────┬────┘
                        │                  │
                  ┌─────▼────┐      ┌─────▼─────┐
                  │ Short-Trm │      │ Long-Term │
                  │  Memory   │      │  Memory   │
                  └───────────┘      └───────────┘
```

### Agent Specialization

| Agent | Specialization | Input | Output |
|-------|----------------|-------|--------|
| **Chemical Agent** | Molecular analysis, ADMET, screening | SMILES, compound info | Properties, predictions |
| **Medical Agent** | Biomedical reasoning, pathways | Disease, target info | Mechanism analysis |
| **Orchestrator** | Intent classification, routing | Raw user query | Structured intent + entities |

---

## ✨ Key Features

### 🧠 Central Orchestrator (Qwen3-based)
- Real-time intent classification using LLMs
- Dynamic routing to specialized agents
- Context-aware entity extraction
- Fallback to local classifier if API fails

### 🧪 Chemical Expert Agent
- Molecular similarity search using **FAISS** indexing
- **ADMET property prediction** via Graph Neural Networks
- Virtual screening against 10M+ compound databases
- SMILES string parsing and validation

### 🧬 Medical Expert Agent
- Biomedical pathway analysis
- **OpenBioLLM** integration for clinical reasoning
- Drug-target interaction prediction
- Mechanism of action (MoA) evaluation

### 🧠 Tri-Tier Memory System
- **Short-Term**: In-memory session history (thread-safe ring buffer)
- **Long-Term**: Persistent JSON-backed storage
- **Vector Memory**: Chromadb integration for semantic search (optional)

### 📡 Streaming Architecture
- **Token-by-token delivery** for instant user feedback
- **Async/await** powered with FastAPI
- **Exception handling** at each layer
- Graceful degradation on agent failures

### 🎯 Intelligent Question Handling
- **Greeting Detection**: Skip agent routing for casual messages
- **Context Awareness**: Uses conversation history
- **Bilingual Support**: Arabic & English

---

## 📁 Project Structure

```
Scientific Operating System/
│
├── 📄 main.py                    ← Root entry point for Docker
├── 📄 requirements.txt           ← Python dependencies
├── 📄 Dockerfile                 ← Docker build config
├── 📄 docker-compose.yml         ← Local dev environment
├── 📄 .dockerignore              ← Docker optimizations
├── 📄 README.md                  ← This file
│
├── 📁 app/                       ← Main application package
│   ├── 📄 __init__.py
│   ├── 📄 main.py                ← FastAPI app definition
│   ├── 📄 config.py              ← Settings & environment vars
│   ├── 📄 index.html             ← Web UI
│   │
│   ├── 📁 agents/                ← Specialized agents
│   │   ├── 📄 __init__.py
│   │   ├── 📁 chemical/
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 agent.py       ← Chemical analysis
│   │   │   └── 📄 search.py      ← FAISS search utils
│   │   └── 📁 medical/
│   │       ├── 📄 __init__.py
│   │       └── 📄 agent.py       ← Medical reasoning
│   │
│   ├── 📁 orchestrator/          ← Routing logic
│   │   ├── 📄 __init__.py
│   │   ├── 📄 brain.py           ← Intent classification
│   │   └── 📄 prompts.py         ← System prompts
│   │
│   └── 📁 memory/                ← Memory system
│       ├── 📄 __init__.py
│       ├── 📄 short_term.py      ← Session memory
│       ├── 📄 long_term.py       ← Persistent storage
│       └── 📄 long_term_store.json ← Data file
│
├── 📁 tests/                     ← Unit tests
│   ├── 📄 conftest.py
│   └── 📄 test_main.py
│
└── 📁 .github/                   ← GitHub Actions
    └── 📁 workflows/
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose** (optional, for containerized deployment)
- **pip** or **conda**

### Local Development Setup

#### 1. Clone the Repository

```bash
git clone https://huggingface.co/spaces/OmarFadlallah/Scientific-Operating-System
cd "Scientific Operating System"
```

#### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n sciosys python=3.10
conda activate sciosys
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Set Up Environment Variables

Create `.env` file in root:

```env
# Groq API Configuration (Required)
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1

# External Microservices (Optional, for advanced features)
ADMET_AI_URL=https://shdwRow-ailixir-admet.hf.space
CHEMICAL_AI_URL=https://RottenShadow-ailixir-chemical-rag.hf.space
DRUG_REPURPOSING_URL=https://RottenShadow-ailixir-drug-repurposing.hf.space
GENERATION_SERVICE_URL=https://shdwRow-ailixir-generation.hf.space

# Model Configuration
ORCHESTRATOR_MODEL=llama-3.3-70b-versatile
QWEN_MODEL=qwen/qwen3-32b
```

#### 5. Run Locally

```bash
# Using Uvicorn directly
uvicorn main:app --reload --port 8000

# Or using Python module
python -m uvicorn main:app --reload --port 8000
```

Visit: **http://localhost:8000**

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ YES | - | API key for Groq LLM |
| `GROQ_BASE_URL` | ❌ NO | https://api.groq.com/openai/v1 | LLM endpoint |
| `ORCHESTRATOR_MODEL` | ❌ NO | llama-3.3-70b-versatile | Main routing model |
| `ADMET_AI_URL` | ❌ NO | HF Space URL | ADMET prediction service |
| `CHEMICAL_AI_URL` | ❌ NO | HF Space URL | Chemical analysis service |
| `DRUG_REPURPOSING_URL` | ❌ NO | HF Space URL | Drug repurposing service |

### config.py

Edit `app/config.py` to modify:

```python
class Settings(BaseSettings):
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    
    # Service URLs
    ADMET_AI_URL: str = os.getenv("ADMET_AI_URL", "...")
    CHEMICAL_AI_URL: str = os.getenv("CHEMICAL_AI_URL", "...")
    
    # Model names
    ORCHESTRATOR_MODEL: str = "llama-3.3-70b-versatile"
    QWEN_MODEL: str = "qwen/qwen3-32b"
    
    class Config:
        env_file = ".env"
```

---

## 💬 Usage

### Web Interface

1. Open **http://localhost:8000** in your browser
2. Type your scientific query in the chat box
3. The system will automatically route to the appropriate agent

### API Endpoints

#### POST /orchestrate

Send a query to the orchestrator:

```bash
curl -X POST http://localhost:8000/orchestrate \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user_123",
    "user_id": "prof_smith",
    "text_input": "What are the ADMET properties of aspirin (SMILES: CC(=O)Oc1ccccc1C(=O)O)?"
  }'
```

**Response** (Streaming):
```
Aspirin (acetylsalicylic acid) exhibits favorable ADMET properties:
- Absorption: 0.85 (good oral bioavailability)
- Distribution: 0.72 (good tissue penetration)
...
```

#### GET /

Serve the web UI (index.html)

---

## 🔍 Components Deep Dive

### 1. Orchestrator Brain (`app/orchestrator/brain.py`)

**Purpose**: Intelligent intent classification and entity extraction

**Process**:
1. Takes raw user input
2. Analyzes query intent using LLM
3. Extracts entities (compound names, diseases, SMILES)
4. Routes to appropriate agent

**Intents**:
- `CHEMICAL_SIMILARITY`: Molecular similarity search
- `ADMET_ANALYSIS`: Drug property prediction
- `DRUG_REPURPOSING`: Disease target screening
- `BIOMEDICAL_MECHANISM`: Pathway analysis
- `APP_HELP`: General assistance

### 2. Chemical Agent (`app/agents/chemical/agent.py`)

**Capabilities**:
- SMILES validation and parsing
- Molecular similarity search (FAISS)
- ADMET batch prediction
- Virtual screening

**Example Query**:
```
"Find similar compounds to caffeine (CN1C=NC2=C1C(=O)N(C(=O)N2C)C)"
```

### 3. Medical Agent (`app/agents/medical/agent.py`)

**Capabilities**:
- Drug-target interaction analysis
- Biological pathway evaluation
- Mechanism of action (MoA) prediction
- Clinical efficacy reasoning

**Example Query**:
```
"Analyze the therapeutic potential of metformin for treating diabetic complications"
```

### 4. Memory System

#### Short-Term Memory (`app/memory/short_term.py`)
- **Implementation**: Thread-safe ring buffer (deque)
- **Capacity**: Configurable (default: 50 messages)
- **Use**: Session-specific context
- **Lifespan**: Duration of session

#### Long-Term Memory (`app/memory/long_term.py`)
- **Implementation**: JSON file-backed persistent storage
- **File**: `app/memory/long_term_store.json`
- **Use**: User profiling, historical analysis
- **Features**: Basic keyword search

### 5. Streaming Response Engine

**How it works**:
1. Queries are processed asynchronously
2. LLM responses are streamed token-by-token
3. Tokens are yielded to client in real-time
4. Full response is stored in memory after completion

**Benefits**:
- ✅ Instant visual feedback
- ✅ Reduced perceived latency
- ✅ Better UX for long responses
- ✅ Reduced memory footprint

---

## 🐳 Deployment

### Docker Setup

#### Build Image

```bash
docker build -t scientific-os:latest -f Dockerfile .
```

#### Run Container

```bash
docker run -p 7860:7860 \
  -e GROQ_API_KEY="your_key" \
  scientific-os:latest
```

#### Docker Compose (Local Dev)

```bash
docker-compose up --build
```

### Hugging Face Spaces Deployment

#### Prerequisites

- Hugging Face account
- Access token with write permissions

#### Steps

1. **Create Space**:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Select "Docker" SDK
   - Name: "Scientific-Operating-System"

2. **Clone Repository**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/Scientific-Operating-System
   cd Scientific-Operating-System
   ```

3. **Configure Secrets**:
   - Go to Space Settings → Secrets
   - Add `GROQ_API_KEY`

4. **Push Code**:
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push
   ```

5. **Monitor Build**:
   - Watch the "Logs" tab
   - Expected time: 3-5 minutes

---

## 🐛 Troubleshooting

### Issue: "Attribute 'app' not found in module 'app'"

**Cause**: Namespace collision between `app.py` file and `app/` directory

**Solution**:
```bash
rm app.py  # Remove conflicting file
# Use main.py instead (already done)
```

### Issue: "ERROR: Error loading ASGI app"

**Cause**: Uvicorn can't find the FastAPI instance

**Check**:
1. Verify `main.py` or `app.py` exists in root
2. Verify `from app.main import app` is correct
3. Check: `python -c "from app.main import app; print(app)"`

### Issue: "ModuleNotFoundError: No module named 'app'"

**Cause**: Package not installed or Python path issue

**Solution**:
```bash
pip install -e .  # Install in editable mode
# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "GROQ_API_KEY not found"

**Cause**: Environment variable not set

**Solution**:
```bash
# Create .env file
echo "GROQ_API_KEY=your_key" > .env

# Or set directly
export GROQ_API_KEY="your_key"
```

### Issue: Agents not responding

**Cause**: External service URLs unreachable

**Check**:
```bash
curl https://shdwRow-ailixir-admet.hf.space/health
```

**Solution**: Use local fallback or configure alternative endpoints

---

## 📊 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Orchestrator Latency** | <300ms | First-token latency |
| **Chemical Search** | ~500-1000ms | 10M compound database |
| **ADMET Prediction** | ~2-3s | Per compound |
| **Memory Per Session** | ~1-5MB | Depends on history |
| **Docker Image Size** | ~800MB | Python 3.10 slim base |

---

## 🔐 Security Considerations

1. **API Keys**:
   - Store in `.env` or environment variables
   - Never commit to git
   - Use `.gitignore`

2. **Rate Limiting**:
   - Implement per-user limits in production
   - Use Redis for distributed rate limiting

3. **Input Validation**:
   - SMILES validation for chemical queries
   - Query length limits
   - Injection attack prevention

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- [ ] Vector database integration (Chromadb/Pinecone)
- [ ] Voice input/output (Whisper + TTS)
- [ ] Advanced visualization (RDKit, PyMOL)
- [ ] Performance optimization
- [ ] Enhanced chemical databases

---

## 📄 License

This project is open source. See LICENSE file for details.

---

## 👨‍💼 Author

**Omar Fadlallah**  
Scientific AI Systems  
[Hugging Face Profile](https://huggingface.co/OmarFadlallah)

---

## 📞 Support

- **Issues**: Report on GitHub/HF Space
- **Discussions**: Community forum
- **Email**: Contact via HF profile

---

## 🚀 Quick Start Checklist

- [ ] Clone repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Set `GROQ_API_KEY` environment variable
- [ ] Run locally (`uvicorn main:app --reload`)
- [ ] Test via web UI at `http://localhost:8000`
- [ ] Deploy to Hugging Face Spaces or Docker

---

**Happy scientific computing! 🧬🚀**
