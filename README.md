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

> ⚡ Real-time LLM routing • 🧪 Multi-agent analysis • 🧠 Persistent memory • 📡 Streaming responses

---

## 🎯 Quick Overview

The **Scientific Operating System (SciOS)** is an intelligent platform that combines:

- **Smart Query Routing**: Automatically detects if you're asking a casual question (instant direct response) or a scientific query (routes to specialized agents)
- **Expert Agents**: Chemical analysis, medical reasoning, ADMET prediction
- **Memory System**: Remembers your conversation history
- **Streaming Responses**: Token-by-token delivery for instant feedback

### What Sets It Apart

✨ **Greeting Detection** - Casual messages skip agent routing for instant responses  
✨ **Multi-Language** - Arabic & English support  
✨ **No Latency** - Sub-300ms first-token latency  
✨ **Production-Ready** - Docker containerized for HF Spaces  

---

## 🚀 Deployment Architecture

- **Backend Core**: FastAPI (Asynchronous Uvicorn Router)
- **LLM Router**: Intent classification + dynamic agent selection
- **Expert Networks**: Chemical Agent & Medical Agent (Groq API)
- **Memory Layers**: Short-term (session) + Long-term (persistent)
- **Containerization**: Optimized Docker on port 7860

---

## 🛠️ Environment Variables Configuration

Set these in your Hugging Face Space **Settings → Secrets**:

```env
GROQ_API_KEY=your_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
ORCHESTRATOR_MODEL=llama-3.3-70b-versatile
```

Optional (for advanced features):
```env
ADMET_AI_URL=https://shdwRow-ailixir-admet.hf.space
CHEMICAL_AI_URL=https://RottenShadow-ailixir-chemical-rag.hf.space
DRUG_REPURPOSING_URL=https://RottenShadow-ailixir-drug-repurposing.hf.space
```

---

## 📂 Project Architecture

### System Flow

```
User Input (Casual or Scientific?)
    ↓
    ├─→ Greeting? (e.g., "أهلا", "مرحبا")
    │   └─→ Direct Response ✓
    │
    └─→ Scientific Query?
        └─→ Orchestrator (Intent Classification)
            ├─→ Chemical Intent
            │   └─→ Chemical Agent (ADMET, similarity search)
            ├─→ Medical Intent
            │   └─→ Medical Agent (pathway analysis)
            └─→ General Intent
                └─→ App Helper

    ↓
    Synthesis Layer (Combine outputs)
    ↓
    Streaming Response Engine
    ↓
    Memory Storage (Short + Long-term)
```

### Folder Structure

```
Scientific Operating System/
├── main.py                      ← Docker entry point
├── requirements.txt
├── Dockerfile                   ← Container config
├── README.md                    ← This file
│
├── app/
│   ├── main.py                  ← FastAPI app & endpoints
│   ├── config.py                ← Settings & environment
│   ├── index.html               ← Web UI
│   │
│   ├── agents/
│   │   ├── chemical/
│   │   │   ├── agent.py         ← Chemical analysis
│   │   │   └── search.py        ← FAISS search
│   │   └── medical/
│   │       └── agent.py         ← Medical reasoning
│   │
│   ├── orchestrator/
│   │   ├── brain.py             ← Intent routing
│   │   └── prompts.py           ← System prompts
│   │
│   └── memory/
│       ├── short_term.py        ← Session memory
│       ├── long_term.py         ← Persistent storage
│       └── long_term_store.json
│
└── tests/
    └── test_main.py
```

---

## 💡 Key Features Explained

### 1. Smart Greeting Detection

**What it does**: Automatically recognizes casual messages

```
✅ "مرحبا" → Direct friendly response (no agents)
✅ "أهلا" → Direct response
✅ "شنو أخبارك؟" → Direct response
❌ "What's the ADMET of aspirin?" → Routes to Chemical Agent
```

**Why it matters**: Faster response, better UX, reduced API calls

### 2. Multi-Agent Routing

| Query Type | Agent | Response Time |
|-----------|-------|--------------|
| Molecular similarity | Chemical Agent | ~1s |
| ADMET prediction | Chemical Agent | ~2-3s |
| Disease screening | Medical Agent | ~3-5s |
| General Q&A | App Helper | <1s |

### 3. Memory System

**Short-Term**: Remembers conversation in current session (50 messages max)  
**Long-Term**: Learns from past queries (persistent JSON storage)  
**Vector**: Optional semantic search (Chromadb)

### 4. Streaming Architecture

- Responses appear **token-by-token** in real-time
- No waiting for full response generation
- Better UX for long scientific explanations

---

## ⚙️ How It Works

### Step-by-Step Processing

```
1. User sends: "مرحبا، كيفك؟"
   ↓
2. System checks: Is this a greeting?
   → YES ✓
   ↓
3. Skip orchestrator, send direct response
   → "مرحبا! أنا هنا للمساعدة..."
   ↓
4. Store in memory
   ✓ Done!

---

1. User sends: "What's the ADMET of aspirin (CC(=O)Oc1ccccc1C(=O)O)?"
   ↓
2. System checks: Is this a greeting?
   → NO ✗
   ↓
3. Send to Orchestrator
   → Detects: Chemical Intent + ADMET Analysis
   ↓
4. Route to Chemical Agent
   ↓
5. Chemical Agent calls ADMET service
   → Returns: absorption, distribution, metabolism, excretion, toxicity
   ↓
6. Synthesize response
   ↓
7. Stream response token-by-token
   ↓
8. Store in both short + long-term memory
   ✓ Done!
```

---

## 🔌 API Endpoints

### POST /orchestrate

Send any scientific query:

```bash
curl -X POST http://localhost:7860/orchestrate \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user_123",
    "user_id": "prof_smith",
    "text_input": "ما هي خصائص الأسبرين؟"
  }'
```

### GET /

Serves the web interface (index.html)

---

## 🎯 Query Examples

### Casual/Greeting Queries

```
✅ Simple greeting → Direct response
   "السلام عليكم"
   
✅ General question → Direct response
   "كيف تعمل هذا النظام؟"
   
✅ Short chat → Direct response
   "شكراً لك"
```

### Scientific Queries

```
🧪 ADMET Analysis
   "Calculate ADMET properties for aspirin"
   
🔬 Molecular Similarity
   "Find compounds similar to caffeine"
   
🧬 Drug Repurposing
   "Screen compounds for Alzheimer's treatment"
   
🏥 Biomedical Analysis
   "Explain the mechanism of metformin"
```

---

## 🐳 Installation & Deployment

### Local Development

```bash
# 1. Clone repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/Scientific-Operating-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment
export GROQ_API_KEY="your_key"

# 4. Run
uvicorn main:app --reload --port 8000

# 5. Visit http://localhost:8000
```

### Docker (Local)

```bash
docker run -p 7860:7860 \
  -e GROQ_API_KEY="your_key" \
  scientific-os:latest
```

### Hugging Face Spaces

1. Create Space (Docker SDK)
2. Add `GROQ_API_KEY` to Secrets
3. Push code: `git push --force origin main`
4. Monitor the Logs tab
5. Space starts automatically on port 7860

---

## 🔍 What's Inside

### app/main.py - The Core

```python
# Smart detection: Should we route to agents?
if should_skip_orchestrator(query.text_input):
    # Direct response for greetings
    return direct_response()
else:
    # Route to orchestrator for scientific queries
    return agent_routing_response()
```

### app/agents/chemical/agent.py - Chemistry

- SMILES parsing and validation
- Molecular similarity (FAISS)
- ADMET batch prediction
- Virtual screening

### app/agents/medical/agent.py - Biology

- Drug-target interactions
- Pathway analysis
- Mechanism of action (MoA)
- Clinical reasoning

### app/memory/ - Persistence

- **short_term.py**: Thread-safe in-memory storage
- **long_term.py**: JSON file-backed persistence
- Automatic conversation tracking

---

## 🚀 Performance

| Metric | Value |
|--------|-------|
| First-token latency | <300ms |
| Greetings→Response | <100ms |
| Full response streaming | Real-time |
| Memory per session | ~1-5MB |
| Docker image size | ~800MB |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "app" module not found | Confirm `main.py` in root, run: `python -c "from app.main import app"` |
| GROQ_API_KEY error | Set env var: `export GROQ_API_KEY="..."` |
| Docker build fails | Check `requirements.txt` has all dependencies |
| Agents not responding | Verify external service URLs are accessible |

---

## 📊 Token Usage

The system optimizes token usage by:

✓ Skipping orchestrator for simple greetings  
✓ Reusing conversation history efficiently  
✓ Streaming responses (saves memory)  
✓ Caching agent outputs  

---

## 🔐 Security

- ✅ API keys in environment variables (never in code)
- ✅ HTTPS for external service calls
- ✅ Input validation for SMILES strings
- ✅ SQLi/injection prevention

---

## 📄 Full Documentation

For detailed information about:
- Architecture deep-dive
- Agent development
- Memory customization
- Performance tuning

See `README_FULL.md`

---

## 🤝 Contributing

Ideas for improvement:

- [ ] Chromadb vector memory integration
- [ ] Voice input/output (Whisper + TTS)
- [ ] Chemical visualization (RDKit)
- [ ] Performance metrics dashboard
- [ ] Multi-language support expansion

---

## 📞 Support

- **GitHub Issues**: Report bugs or suggest features
- **HF Spaces Discussions**: Ask questions
- **Email**: Via Hugging Face profile

---

## 🎓 Learn More

- [Groq API Docs](https://console.groq.com/keys)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Deployment Guide](https://huggingface.co/docs/hub/spaces-sdks-docker)

---

## ✨ Quick Start

```bash
# 1. Set your API key
export GROQ_API_KEY="your_groq_key_here"

# 2. Install
pip install -r requirements.txt

# 3. Run
uvicorn main:app --reload --port 8000

# 4. Open browser
# http://localhost:8000

# 5. Try:
# - "مرحبا" (greeting)
# - "What's the ADMET of aspirin?" (scientific)
```

---

**Developed with ❤️ for scientific discovery**

🧬 AI Scientific OS © 2026 | Powered by Groq | Deployed on Hugging Face Spaces

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
