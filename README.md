---
title: Scientific Operating System
emoji: 🧬
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 🧬 Scientific Operating System — Complete Technical Documentation

> **AI-lixir**: A multi-agent AI Scientific Operating System built for Drug Discovery, Cheminformatics, and Biomedical Research. Deployed on Hugging Face Spaces via Docker, powered by Groq LLM inference.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Configuration & Environment Variables](#4-configuration--environment-variables)

5. [Startup & Lifespan](#5-startup--lifespan)

6. [Orchestrator Brain](#6-orchestrator-brain)

7. [Request Routing Pipeline](#7-request-routing-pipeline)

8. [Agents](#8-agents)

   * 8.1 [Chemical Agent](#81-chemical-agent)
   * 8.2 [Medical Agent](#82-medical-agent)
   * 8.3 [RAG Agent (Customer Support)](#83-rag-agent-customer-support)
   * 8.4 [APP Agent (Conversational)](#84-app-agent-conversational)

9. [RAG Pipeline (ai-lixir-rag-system)](#9-rag-pipeline-ai-lixir-rag-system)

   * 9.1 [Configuration (src/config.py)](#91-configuration-srcconfigpy)
   * 9.2 [Embeddings (src/embeddings.py)](#92-embeddings-srcembeddingspy)
   * 9.3 [Indexer (src/indexer.py)](#93-indexer-srcindexerpy)
   * 9.4 [Ingestion Service (src/ingestion_service.py)](#94-ingestion-service-srcingestion_servicepy)
   * 9.5 [Engine Builder (src/engine.py)](#95-engine-builder-srcenginepy)
   * 9.6 [Chunking Strategies](#96-chunking-strategies)

10. [Memory System](#10-memory-system)

    * 10.1 [Short-Term Memory](#101-short-term-memory)
    * 10.2 [Long-Term Memory](#102-long-term-memory)

11. [Audio Pipeline](#11-audio-pipeline)

    * 11.1 [Speech-to-Text (STT)](#111-speech-to-text-stt)
    * 11.2 [Text-to-Speech (TTS)](#112-text-to-speech-tts)
    * 11.3 [Voice Activity Detection (VAD)](#113-voice-activity-detection-vad)

12. [API Reference](#12-api-reference)

    * 12.1 [HTTP Endpoints](#121-http-endpoints)
    * 12.2 [WebSocket Voice Channel](#122-websocket-voice-channel)

13. [Monitoring System](#13-monitoring-system)

    * 13.1 [Overview](#131-overview)
    * 13.2 [app/monitoring.py](#132-appmonitoringpy)
    * 13.3 [Dashboard (GET /monitor)](#133-dashboard-get-monitor)
    * 13.4 [Metrics API](#134-metrics-api)

14. [CI/CD Pipeline](#14-cicd-pipeline)

    * 14.1 [Overview](#141-overview)
    * 14.2 [Workflow File](#142-workflow-file)
    * 14.3 [GitHub Secret Setup](#143-github-secret-setup)
    * 14.4 [Manual Trigger](#144-manual-trigger)

15. [Data Persistence on HF Spaces](#15-data-persistence-on-hf-spaces)

16. [Dependency Matrix](#16-dependency-matrix)

17. [Known Bugs & Fixes Applied](#17-known-bugs--fixes-applied)

18. [Deployment Guide (HF Spaces)](#18-deployment-guide-hf-spaces)


---

## 1. System Overview

Scientific Operating System is a **multi-agent AI backend** that acts as a centralized scientific intelligence platform. It receives natural language queries (text or voice) and routes them to the appropriate specialist agent:

```
User Input (text / voice)
        │
        ▼
  Orchestrator Brain
  (intent classification)
        │
   ┌────┴────────────────────────┐
   │                             │
Chemical Agent            Medical Agent
(ADMET, SMILES,        (Biomedical reasoning,
 Similarity Search,      Drug-Target Interaction,
 Drug Repurposing)       Pathway Analysis)
   │                             │
   └────────────┬────────────────┘
                │
           RAG Agent
    (Documentation Q&A
     via Vector Search)
                │
           APP Agent
    (Greetings, General
     Conversational AI)
```

**Core Technology Stack:**

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| LLM Provider | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace (`intfloat/multilingual-e5-large-instruct`) |
| Vector Store | Weaviate (primary) / In-Memory + Disk (fallback) |
| STT | Groq Whisper (`whisper-large-v3-turbo`) |
| TTS | OpenAI TTS-1 (optional) |
| Memory | Redis (primary) / JSON file (fallback) |
| RAG Framework | LlamaIndex 0.10.x |
| Deployment | Docker on Hugging Face Spaces |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HF Spaces Docker Container                   │
│                                                                     │
│  ┌──────────────┐    ┌───────────────────────────────────────────┐ │
│  │   FastAPI     │    │              app/main.py                  │ │
│  │   app:7860    │◄───│  Routes: /orchestrate  /audio/*  /rag/*  │ │
│  │               │    │          /ws/voice  /health  /           │ │
│  └──────────────┘    └───────────────────────────────────────────┘ │
│          │                                                          │
│          ▼                                                          │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    route_and_stream()                         │ │
│  │  Fast Path: greetings → 1 LLM call                           │ │
│  │  Slow Path: science  → routing call + agent(s) + synthesis   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│          │                                                          │
│    ┌─────┴──────┬────────────┬──────────────┐                      │
│    ▼            ▼            ▼              ▼                      │
│ Chemical     Medical       RAG           APP                       │
│  Agent        Agent       Agent         Agent                      │
│ (httpx→       (Groq       (LlamaIndex   (Groq                      │
│  HF Spaces)   LLM)         +Weaviate)   LLM)                       │
│    │                         │                                      │
│    ▼                         ▼                                      │
│ External HF Spaces     /data/rag_index  (disk)                     │
│  - ADMET AI             _GLOBAL_IN_MEMORY_INDEX  (RAM)             │
│  - Chemical RAG                                                     │
│  - Drug Repurposing                                                 │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐  │
│  │ ShortTermMem │    │            LongTermMemory                │  │
│  │  (RAM deque) │    │  Redis ──► /code/app/memory/*.json  ⚠️  │  │
│  └──────────────┘    └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

> ⚠️ **Note:** The JSON fallback path `/code/app/memory/long_term_store.json` is inside the ephemeral container filesystem. See [Section 13](#13-data-persistence-on-hf-spaces) for persistence details.

---

## 3. Project Structure

```
Scientific-Operating-System/
│
├── app/
│   ├── __init__.py
│   ├── config.py                    # Pydantic settings — all env vars
│   ├── main.py                      # FastAPI app, routes, lifespan, WebSocket
│   ├── audio.py                     # AudioProcessor: STT (Groq) + TTS (OpenAI)
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── brain.py                 # OrchestratorBrain: intent classification
│   │   └── prompts.py               # All LLM system prompts
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── voice_wrapper.py         # VoiceEnabledAgent: wraps any agent with TTS
│   │   ├── chemical/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py             # ChemicalAgent: ADMET, similarity, repurposing
│   │   │   └── search.py            # (FAISS placeholder)
│   │   ├── medical/
│   │   │   └── agent.py             # MedicalAgent: biomedical LLM reasoning
│   │   └── customer_support/
│   │       ├── __init__.py
│   │       ├── agent.py             # CustomerSupportRAGAgent: RAG singleton
│   │       └── ai-lixir-rag-system/
│   │           ├── main.py          # Standalone entry point
│   │           ├── chunking/
│   │           │   ├── base.py      # BaseChunkingStrategy ABC
│   │           │   ├── factory.py   # ChunkingFactory
│   │           │   └── strategies.py # Markdown, Sentence, Token strategies
│   │           └── src/
│   │               ├── config.py    # RAG-specific config (dual-mode)
│   │               ├── embeddings.py # EmbeddingProviderFactory + E5InstructEmbedding
│   │               ├── engine.py    # RAGEngineBuilder: hybrid query engine
│   │               ├── indexer.py   # VectorIndexManager: Weaviate + disk + RAM
│   │               └── ingestion_service.py # RAGIngestionService: file → nodes → index
│   │
│   └── memory/
│       ├── __init__.py
│       ├── short_term.py            # ShortTermMemory: in-RAM ring buffer
│       └── long_term.py             # LongTermMemory: Redis + JSON fallback
│
├── tests/
│   └── test_main.py
│
├── requirements.txt                 # Pinned dependency versions
├── Dockerfile                       # HF Spaces Docker build
├── docker-compose.yml               # Local dev compose
└── .env.example                     # Environment variable template
```

---

## 4. Configuration & Environment Variables

**File:** `app/config.py`

All configuration is managed via Pydantic `BaseSettings`, which reads from environment variables or a `.env` file.

```python
class Settings(BaseSettings):
    GROQ_API_KEY: str            # REQUIRED — Groq API key (LLM + Whisper STT)
    GROQ_BASE_URL: str           # Default: https://api.groq.com/openai/v1
    GROQ_WHISPER_MODEL: str      # Default: whisper-large-v3-turbo
    
    EMBEDDING_PROVIDER: str      # Default: huggingface
    EMBEDDING_MODEL: str         # Default: intfloat/multilingual-e5-large-instruct
    
    OPENAI_API_KEY: str          # Optional — only needed for OpenAI TTS
    
    # External microservices (other HF Spaces)
    ADMET_AI_URL: str            # ADMET property prediction service
    CHEMICAL_AI_URL: str         # Chemical similarity & RAG service
    DRUG_REPURPOSING_URL: str    # Virtual screening service
    GENERATION_SERVICE_URL: str  # Generation utility service
    
    ORCHESTRATOR_MODEL: str      # Default: llama-3.3-70b-versatile
    QWEN_MODEL: str              # Default: qwen/qwen3-32b (fallback classifier)
    
    # Redis (optional — JSON fallback if unavailable)
    REDIS_HOST: str              # Default: localhost
    REDIS_PORT: int              # Default: 6379
    REDIS_DB: int                # Default: 0
    
    # Weaviate (optional — in-memory + disk fallback if unavailable)
    WEAVIATE_HOST: str           # Default: localhost
    WEAVIATE_PORT: int           # Default: 8080
    WEAVIATE_GRPC_PORT: int      # Default: 50051
```

**Minimum required for HF Spaces deployment:**

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

**Optional for full feature set:**

```env
OPENAI_API_KEY=sk-xxxx          # Enables high-quality TTS
EMBEDDING_PROVIDER=huggingface  # huggingface | openai
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
```

---

## 5. Startup & Lifespan

**File:** `app/main.py` — `lifespan()` async context manager

The app uses FastAPI's lifespan hook for all startup/shutdown logic. Executed **once** at container boot:

```
Startup Sequence
════════════════
1. _init_redis()
   ├── Attempts Redis connection (0.5s timeout)
   └── Sets redis_available = True/False

2. LongTermMemory initialization
   ├── If Redis available → Redis-backed store
   └── If Redis unavailable → JSON file at /code/app/memory/long_term_store.json
                              ⚠️ Ephemeral on HF Spaces!

3. RQ Worker (only if Redis available)
   └── subprocess.Popen([python -m rq worker default])
       Background job queue for heavy ingestion tasks

4. asyncio.create_task(rag_agent._initialise())
   └── Non-blocking background task — system accepts requests
       before this completes (ReadinessMiddleware gates other routes)

Shutdown Sequence
═════════════════
1. worker_process.terminate() — kill RQ worker
2. rag_agent.close()          — close Weaviate connection
```

**ReadinessMiddleware:**

A custom Starlette middleware that returns HTTP 503 for all routes except:
- `/` (web UI)
- `/rag/status`
- `/health`
- `/docs`, `/openapi.json`, `/redoc`
- `/rag/ingest/status/{job_id}`

This prevents requests from hitting uninitialized agents during the cold start.

---

## 6. Orchestrator Brain

**File:** `app/orchestrator/brain.py`

`OrchestratorBrain` is a fallback keyword classifier used **only** when the primary Groq-based combined routing in `route_and_stream()` fails.

```python
class OrchestratorBrain:
    def classify_intent(self, query: str) -> dict
    def _fallback_classify(self, query: str) -> dict  # keyword-based
```

**Primary classifier** (in `main.py`) is a single Groq LLM call with `response_format={"type": "json_object"}` using `COMBINED_ORCHESTRATOR_PROMPT`. This prompt classifies intent AND routes in one API call:

```json
{
  "intent": "CHEMICAL_SIMILARITY|ADMET_ANALYSIS|DRUG_REPURPOSING|BIOMEDICAL_MECHANISM|APP_SUPPORT_RAG|APP_HELP|OUT_OF_DOMAIN",
  "target_agent": "CHEMICAL_AGENT|MEDICAL_AGENT|RAG_AGENT|APP_AGENT|NONE",
  "entities": {"compound": "", "smiles": "", "disease": ""},
  "out_of_domain_reason": ""
}
```

**Fallback classifier** (keyword-based) triggers when the Groq call fails:

```
Keyword Groups
══════════════
Greetings    → app_agent (hello, hi, شكرا, مرحبا, …)
Chemical     → chemical  (smiles, admet, molecule, compound, …)
Medical      → medical   (disease, pathway, protein, receptor, …)
Default      → app_agent
```

**System prompts** are centralized in `app/orchestrator/prompts.py`:
- `ORCHESTRATOR_SYSTEM_PROMPT` — for `OrchestratorBrain.classify_intent()`
- `CHEMICAL_AGENT_SYSTEM_PROMPT`
- `MEDICAL_AGENT_SYSTEM_PROMPT`
- `APP_SUPPORT_RAG_SYSTEM_PROMPT`
- `APP_AGENT_SYSTEM_PROMPT`

---

## 7. Request Routing Pipeline

**File:** `app/main.py` — `route_and_stream()`

This is the core function handling **all** query routing. It's an async generator that yields text tokens for streaming.

```
route_and_stream(text_input, session_id, user_id)
═════════════════════════════════════════════════

Step 1: Greeting Check
───────────────────────
is_general_greeting(text) ?
  YES → Fast Path:
        1. Get chat history (ShortTermMemory, last 12 messages)
        2. Single Groq streaming call (APP_AGENT personality)
        3. Yield tokens
        4. Save to ShortTermMemory + LongTermMemory
        5. RETURN
  NO  → Continue to Step 2

Step 2: Combined Routing (1 Groq API call)
───────────────────────────────────────────
Groq call → JSON {intent, target_agent, entities, out_of_domain_reason}
  Exception → OrchestratorBrain._fallback_classify() (keyword matching)

Step 3: Domain Check
─────────────────────
intent == OUT_OF_DOMAIN ?
  YES → Emit bilingual refusal message (Arabic/English auto-detected)
        RETURN

Step 4: Parallel Agent Execution
──────────────────────────────────
asyncio.gather(*tasks) — runs simultaneously:
  - CHEMICAL_AGENT task  (if chemical/ADMET/repurposing intent)
  - MEDICAL_AGENT task   (if disease entity + medical intent)
  
  RAG_AGENT runs separately (not in parallel — it's a retrieval, not compute)

Step 5: Synthesis or Direct Streaming
───────────────────────────────────────
RAG output?
  YES → Stream directly word-by-word (8ms delay for natural feel)
        (RAG answers are already grounded, no re-synthesis needed)
  NO  → Build synthesis context:
        "[Chem Data]: ... [Bio Data]: ..."
        Single Groq streaming call (synthesis LLM)
        Yield tokens

Step 6: Memory Persistence
───────────────────────────
ShortTermMemory.add_message() × 2 (user + assistant)
SESSION_MEMORY dict update
LongTermMemory.add_entry() (with intent metadata)
```

**Greeting detection** (`is_general_greeting()`):
Regex patterns covering English and Arabic greetings, farewells, thanks, social phrases. Also catches messages ≤3 words with no scientific keywords.

**Scientific keywords that bypass greeting check:**
`compound, drug, disease, molecule, chemical, smiles, admet, screening, pathway, protein, target, receptor, ligand, inhibitor, biomarker, clinical, genome, dna, rna, enzyme, pharmacology` + Arabic equivalents.

---

## 8. Agents

### 8.1 Chemical Agent

**File:** `app/agents/chemical/agent.py`

Calls **external Hugging Face Spaces microservices** via `httpx`. Timeout: 60 seconds.

```
Intent Routing
══════════════
ADMET_ANALYSIS / toxicity / property / absorption
  └── POST {ADMET_AI_URL}/predict_batch
      Payload: {"smiles_list": ["<SMILES>"]}
      Returns: Absorption, Distribution, Metabolism, Excretion, Toxicity scores

DRUG_REPURPOSING / screen / target
  └── POST {DRUG_REPURPOSING_URL}/api/v1/screen
      Payload: {"disease_name": "...", "min_score": 0, "top_n_targets": 5}
      Returns: Top 3 drug-target candidates with binding scores

CHEMICAL_SIMILARITY (default)
  └── POST {CHEMICAL_AI_URL}/search/full-rag  OR  /search/retrieval-only
      (full-rag if "explain" or "detailed" in intent)
      Payload: {"smiles": "...", "top_k": 3, "explain": true}
      Returns: Top 3 similar compounds with similarity scores
```

**Entity extraction** relies on the Orchestrator's JSON output:
- `entities.compound` — compound name or identifier
- `entities.smiles` — SMILES string
- `entities.disease` — disease name

### 8.2 Medical Agent

**File:** `app/agents/medical/agent.py`

A **pure LLM agent** — no external API calls, no RAG. Uses Groq `llama-3.3-70b-versatile` with `temperature=0.0` for maximum scientific accuracy.

```
System Prompt Focus:
- Molecular Pharmacology & Drug Discovery specialist
- Drug-Target Interactions, receptor binding, signaling pathways
- Genomic/proteomic feasibility for drug repurposing
- NO clinical advice or symptom checklists

Intent Branching:
  repurpose / screen → therapeutic rationale prompt
  else               → ADMET verification context prompt
```

**Input:** `intent` (string) + `entities` dict  
**Output:** `"[Biomedical Reasoning Engine Output]:\n{LLM response}"`

### 8.3 RAG Agent (Customer Support)

**File:** `app/agents/customer_support/agent.py`

The most complex agent — a singleton (`CustomerSupportRAGAgent`) wrapping the full LlamaIndex RAG pipeline.

```
Class: CustomerSupportRAGAgent
══════════════════════════════
Constants:
  INDEX_NAME = "AilixirDocs"   # Weaviate collection name
  TOP_K = 5                    # retrieved chunks per query
  ALPHA = 0.5                  # hybrid search balance (50% vector, 50% BM25)

State:
  _ready: bool                 # True after successful initialization
  _lock: asyncio.Lock          # prevents duplicate init race conditions
  _query_engine                # LlamaIndex query engine (built once)
  _index_manager               # VectorIndexManager instance
  _ingestion_svc               # RAGIngestionService instance

Global State (rag_state dict):
  ready: bool                  # gates ReadinessMiddleware
  embedding_initialized: bool  # prevents re-downloading embedding model
  llm_initialized: bool        # prevents re-initializing Groq LLM
```

**Initialization sequence** (`_initialise()`):
```
1. Acquire asyncio.Lock (prevents concurrent double-init)
2. _bootstrap_llama_index()
   ├── EmbeddingProviderFactory.create_embedding_model()
   │   └── E5InstructEmbedding or HuggingFaceEmbedding
   │       Downloads model ~560MB on first cold start
   └── Groq LLM via llama-index-llms-groq
3. VectorIndexManager(index_name="AilixirDocs")
4. load_persisted_index()
   ├── Try Weaviate → load collection
   ├── Try /data/rag_index → load_from_storage()
   ├── Try _GLOBAL_IN_MEMORY_INDEX → reuse RAM index
   └── No data → mark ready=True, wait for first ingestion
5. RAGEngineBuilder.build_hybrid_query_engine(top_k=5, alpha=0.5)
6. rag_state["ready"] = True
```

**Query execution** (`run()`):
```
1. Ensure initialized (lazy init if not)
2. loop.run_in_executor(None, query_engine.query)
   LlamaIndex is synchronous → must run in thread pool
3. Return str(response).strip()
```

**Reload flow** (`reload_engine()`):
After ingestion, called to pick up new documents:
```
1. Lock acquired
2. _ready = False
3. _query_engine = None
4. rag_state["ready"] = False  ← Bug 1 fix: resets global flag
5. Lock released
6. _initialise() called again
   └── embedding_initialized=True → skips re-download (Bug 6 fix)
```

### 8.4 APP Agent (Conversational)

Not a separate class — handled inline in `route_and_stream()` as the fast path for greetings and the fallback for ambiguous queries. Uses `ORCHESTRATOR_MODEL` with the APP_AGENT personality prompt, temperature=0.7.

---

## 9. RAG Pipeline (ai-lixir-rag-system)

Located at `app/agents/customer_support/ai-lixir-rag-system/`. This is a self-contained sub-package added to `sys.path` at runtime.

### 9.1 Configuration (src/config.py)

Dual-mode config — works both embedded in the app and as a standalone module:

```python
# Mode 1: Embedded (imports from app.config.settings)
from app.config import settings as app_settings
GROQ_API_KEY = app_settings.GROQ_API_KEY
...

# Mode 2: Standalone (reads env vars directly via pydantic)
class _StandaloneSettings(BaseSettings): ...
```

**Embedding dimension auto-detection:**
```python
EMBED_DIM = (
    1024 if "e5-large"         in EMBED_MODEL else  # multilingual-e5-large-instruct
    1024 if "bge-m3"           in EMBED_MODEL else  # BAAI/bge-m3
    1536 if "text-embedding-3" in EMBED_MODEL else  # OpenAI
    768  if "bge-small"        in EMBED_MODEL else  # bge-small
    384                                              # MiniLM (default)
)
```

### 9.2 Embeddings (src/embeddings.py)

**`EmbeddingProviderFactory.create_embedding_model(provider, model_name, **kwargs)`**

Supports three providers:

```
Provider: "groq"
  → Remapped to huggingface (Groq has no embeddings API endpoint)
  → Logs warning and continues with default multilingual model

Provider: "openai"
  → llama_index.embeddings.openai.OpenAIEmbedding
  → Requires api_key kwarg (OPENAI_API_KEY)

Provider: "huggingface" (default)
  ├── If model name contains "e5" AND "instruct"
  │   └── E5InstructEmbedding (custom class, see below)
  └── Else
      └── llama_index.embeddings.huggingface.HuggingFaceEmbedding

Fallback (any exception):
  └── _make_sentence_transformer("paraphrase-multilingual-MiniLM-L12-v2")
```

**`E5InstructEmbedding`** — Custom class for the `intfloat/multilingual-e5-*-instruct` model family:

The e5-instruct models require specific prefix formatting for accurate retrieval:
```
Queries  → "Instruct: {task}\nQuery: {text}"
Documents → no prefix (plain text)
```

Without these prefixes, retrieval quality drops significantly. The class applies them automatically:
```python
def _get_query_embedding(self, query: str) -> List[float]:
    return self._st.encode(
        f"Instruct: {self._task}\nQuery: {query}",
        normalize_embeddings=True
    ).tolist()

def _get_text_embedding(self, text: str) -> List[float]:
    # Documents: no prefix
    return self._st.encode(text, normalize_embeddings=True).tolist()
```

Task description used for Arabic RAG:
```
"Given a user question, retrieve relevant document passages that answer the question"
```

### 9.3 Indexer (src/indexer.py)

**`VectorIndexManager`**

Manages the three-tier persistence strategy:

```
Tier 1: Weaviate (primary, not available on HF Spaces free tier)
  └── weaviate.connect_to_local(host, port, grpc_port)
      Collection: "AilixirDocs"

Tier 2: Disk persistence at /data/rag_index (HF Spaces Persistent Storage)
  └── StorageContext.load_from_storage(persist_dir=PERSIST_DIR)

Tier 3: RAM — _GLOBAL_IN_MEMORY_INDEX (class-level variable)
  └── VectorIndexManager._GLOBAL_IN_MEMORY_INDEX
      Lives for the duration of the server process
      Lost on restart if /data is not available
```

**Persistence directory resolution (Bug 2 fix):**
```python
_HF_PERSISTENT = pathlib.Path("/data/rag_index")
_LOCAL_FALLBACK = _RAG_SYSTEM_ROOT / "storage" / "rag_index"

try:
    # Test /data writability (HF Spaces with Persistent Storage enabled)
    _HF_PERSISTENT.parent.mkdir(parents=True, exist_ok=True)
    _test = _HF_PERSISTENT.parent / ".write_test"
    _test.touch(); _test.unlink()
    PERSIST_DIR = str(_HF_PERSISTENT)      # ✅ /data/rag_index
except Exception:
    PERSIST_DIR = str(_LOCAL_FALLBACK)     # /code/.../storage/rag_index
```

**Compatibility shim (applied at import time):**
```python
# llama-index-core >=0.11 removed TextNode.get_doc_id()
# but llama-index-vector-stores-weaviate 1.x still calls it
if not hasattr(_TextNode, "get_doc_id"):
    _TextNode.get_doc_id = lambda self: self.ref_doc_id or self.node_id
```

**`load_persisted_index()` — waterfall:**
```
1. Try Weaviate
   → weaviate.connect_to_local() + WeaviateVectorStore
   → VectorStoreIndex.from_vector_store()

2. Try disk
   → os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR)
   → StorageContext.from_defaults(persist_dir=PERSIST_DIR)
   → load_index_from_storage()
   → Store result in _GLOBAL_IN_MEMORY_INDEX

3. Try existing RAM index
   → VectorIndexManager._GLOBAL_IN_MEMORY_INDEX is not None
   → Return it

4. All failed → raise Exception (caller handles gracefully)
```

### 9.4 Ingestion Service (src/ingestion_service.py)

**`RAGIngestionService`**

Handles the full document ingestion pipeline:

```
ingest_bytes(filename, content, strategy) or ingest_files(file_paths, strategy)
                          │
                          ▼
              SimpleDirectoryReader.load_data()
                          │
                          ▼
              ChunkingFactory.get_strategy(strategy)
              .chunk(documents) → List[BaseNode]
                          │
                    ┌─────┴─────┐
                    ▼           ▼
              Weaviate       In-Memory / Disk
              available?
                YES           NO
                │              │
                ▼              ▼
         Compatibility    Existing index?
         shim applied     YES → insert nodes (Bug 3 fix)
                │         NO  → build from scratch
                ▼              │
         VectorStoreIndex      ▼
         (Weaviate-backed)  storage_context.persist(PERSIST_DIR)
```

**Additive ingestion (Bug 3 fix):**
Without this fix, every new document upload would **overwrite** the existing index. The fix checks for an existing in-memory index and inserts new nodes instead:
```python
existing = VectorIndexManager._GLOBAL_IN_MEMORY_INDEX
if existing is not None:
    for node in nodes:
        existing.insert(node)   # additive — preserves previous data
    index = existing
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # First ingestion — build fresh
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    storage_context.persist(persist_dir=PERSIST_DIR)
```

**Weaviate connectivity:**
- Lazy connection — only attempts connection on first `_get_vector_store()` call
- Sets `_client_failed = True` on connection failure → never retries during same process lifetime
- Gracefully falls back to disk/RAM

### 9.5 Engine Builder (src/engine.py)

**`RAGEngineBuilder.build_hybrid_query_engine(top_k, alpha)`**

Builds the LlamaIndex query engine with a custom anti-hallucination QA prompt:

```
Custom QA Prompt Template:
"Context information is provided strictly below:
 ---------------------
 {context_str}
 ---------------------
 Given the context information and NOT prior knowledge,
 answer the user query accurately, structurally, and professionally.
 If the answer cannot be found or inferred directly from the provided context,
 honestly state that the information is not available in the documentation.
 Query: {query_str}
 Answer: "
```

**Engine modes:**
```
Weaviate backend → hybrid search mode
  index.as_query_engine(
      vector_store_query_mode="hybrid",
      alpha=0.5,        # 50% vector + 50% BM25
      similarity_top_k=5
  )

In-memory backend → standard vector search
  index.as_query_engine(
      vector_store_query_mode="default",
      similarity_top_k=5
  )
```

### 9.6 Chunking Strategies

**File:** `chunking/strategies.py` — three strategies available:

```
MarkdownStrategy (default)
  └── MarkdownNodeParser.get_nodes_from_documents()
      Splits on markdown headers (##, ###, etc.)
      Best for: structured .md documentation files

SentenceStrategy
  └── SentenceSplitter(chunk_size=512, chunk_overlap=20)
      Splits on sentence boundaries, respects token limits
      Best for: general prose documents

TokenStrategy
  └── TokenTextSplitter(chunk_size=512, chunk_overlap=20)
      Pure token-count splitting, no semantic awareness
      Best for: when strict token control is required
```

**ChunkingFactory:**
```python
ChunkingFactory.get_strategy("markdown")  # → MarkdownStrategy()
ChunkingFactory.get_strategy("sentence", chunk_size=256, chunk_overlap=50)
ChunkingFactory.get_strategy("token", chunk_size=1024, chunk_overlap=0)
```

---

## 10. Memory System

### 10.1 Short-Term Memory

**File:** `app/memory/short_term.py`

Thread-safe in-memory ring buffer. Per-session conversation history.

```python
class ShortTermMemory:
    _store: Dict[str, deque]   # session_id → deque(maxlen=50)
    _lock: Lock                # thread safety

    add_message(session_id, role, content)
    get_history(session_id, limit=10) → List[{"role": str, "content": str}]
    clear(session_id)
```

**Behavior:**
- Max 50 messages per session (ring buffer — oldest auto-evicted)
- `get_history(limit=10)` returns last N messages → fed as context to LLM
- Lost on process restart (intentional — it's "short-term")
- Thread-safe via `threading.Lock`

### 10.2 Long-Term Memory

**File:** `app/memory/long_term.py`

Persistent store with Redis primary and JSON file fallback.

```python
class LongTermMemory:
    path: str         # JSON file path
    is_redis: bool    # True if Redis connection succeeded
    redis_client      # redis.Redis instance
    _store: list      # in-memory cache (JSON fallback only)
    _lock: Lock       # thread safety for JSON writes

    add_entry(session_id, text, metadata)
    search(query, top_k=5) → List[entry]
    _load_json_store()
```

**Storage modes:**

```
Redis mode (is_redis=True):
  add_entry → rpush("long_term_memory", json.dumps(entry))
  search    → lrange("long_term_memory", 0, -1)

JSON mode (is_redis=False):
  path: /code/app/memory/long_term_store.json  ⚠️ EPHEMERAL
  add_entry → reload from disk → append → prune if >5000 → write back
  search    → load from _store (in-memory cache)
```

**Entry structure:**
```json
{
  "session_id": "user_abc_123",
  "text": "User: what is ADMET?\nAssistant: ADMET stands for...",
  "metadata": {
    "intent": "APP_HELP",
    "agent": "APP_AGENT"
  }
}
```

**Search algorithm:** Simple keyword scoring
- Exact substring match: +5 points
- Per-word overlap: +1 point each
- Returns top-k by score

> ⚠️ **HF Spaces Warning:** The JSON file path `/code/app/memory/long_term_store.json` is inside the ephemeral container filesystem. All conversation history is lost on restart/rebuild. To make it persistent, change the path to `/data/long_term_store.json` and enable Persistent Storage in HF Space settings.

---

## 11. Audio Pipeline

**File:** `app/audio.py` — `AudioProcessor` singleton

### 11.1 Speech-to-Text (STT)

Uses **Groq Whisper** (`whisper-large-v3-turbo`). No OpenAI key required.

```python
async def transcribe_audio(audio_file: bytes, audio_format: str = "webm") -> str:
```

**Process:**
1. Wrap bytes in `io.BytesIO` with a filename hint (e.g., `recording.webm`)
2. Call `groq_client.audio.transcriptions.create()`
3. Groq returns plain text when `response_format="text"`
4. Strip whitespace and return

**Prompt hint to Whisper:**
```
"Scientific query in English or Arabic (العربية). Chemistry, biology, compound, SMILES, medicine, research."
```
This improves transcription accuracy for scientific terminology in both languages.

**Supported formats:** webm, mp4, wav, mp3, m4a, ogg, flac

**Chunk concatenation (WebSocket):**
```python
async def transcribe_chunks(chunks: list[bytes], audio_format: str) -> str:
    combined = b"".join(chunks)
    return await transcribe_audio(combined, audio_format)
```

### 11.2 Text-to-Speech (TTS)

Uses **OpenAI TTS-1**. Only available when `OPENAI_API_KEY` is configured. Browser `SpeechSynthesis` API is the free fallback (handled client-side).

```python
async def synthesize_speech(text: str, voice: str = "nova") -> bytes:
    # Returns MP3 bytes
```

**Available voices:** `alloy`, `echo`, `fable`, `nova`, `onyx`, `shimmer`  
**Default:** `nova`

### 11.3 Voice Activity Detection (VAD)

Energy-based VAD — no external ML library required.

```python
@staticmethod
def compute_rms(pcm_bytes: bytes, sample_width: int = 2) -> float:
    # Computes RMS energy of 16-bit PCM audio

@staticmethod
def is_speech(rms: float, threshold: float = 500.0) -> bool:
    # Returns True if RMS > threshold
```

**How it works:**
1. Client sends `{"type": "vad_energy", "rms": 342.5}` messages
2. Server calls `is_speech(rms)` with threshold=500.0
3. Returns `{"type": "vad_status", "speaking": true/false}` to client
4. State change events only (not every frame)

---

## 12. API Reference

### 12.1 HTTP Endpoints

#### `GET /`
Returns the web UI (`app/index.html`).

---

#### `POST /orchestrate`
Main text query endpoint. Streams response as plain text.

**Request body:**
```json
{
  "session_id": "user_session_abc",
  "user_id": "user_123",
  "text_input": "What is the ADMET profile of aspirin?"
}
```

**Response:** `StreamingResponse` (text/plain), token by token.

---

#### `POST /audio/transcribe`
Transcribes audio file to text using Groq Whisper.

**Request:** `multipart/form-data`
- `file`: Audio file (webm, wav, mp3, etc.)
- `audio_format`: Format string (default: "webm")

**Response:**
```json
{
  "status": "success",
  "transcribed_text": "What is the ADMET profile of aspirin?",
  "audio_format": "webm",
  "model": "whisper-large-v3-turbo"
}
```

---

#### `POST /audio/synthesize`
Converts text to speech (requires `OPENAI_API_KEY`).

**Request body:**
```json
{"text": "The ADMET profile of aspirin is...", "voice": "nova"}
```

**Response:** `StreamingResponse` (audio/mpeg), MP3 bytes.

---

#### `POST /audio/agent-voice`
Full voice-to-voice pipeline: STT → Agent → TTS.

**Request:** `multipart/form-data`
- `file`: Audio file
- `session_id`, `user_id`, `audio_format`, `voice`: Form fields

**Response:** MP3 audio if `OPENAI_API_KEY` set, else JSON with text.

---

#### `POST /rag/ingest`
Ingests a Markdown file into the knowledge base. Runs as a background task.

**Request:** `multipart/form-data`
- `file`: `.md` or `.txt` file
- `strategy`: `markdown` | `sentence` | `token` (default: `markdown`)

**Response:**
```json
{
  "status": "success",
  "job_id": "3dbb49a8-06b3-43d7-b9e0-f63ac1ad3b56",
  "filename": "docs.md",
  "strategy": "markdown",
  "message": "Ingestion job scheduled (running in background)."
}
```

---

#### `GET /rag/ingest/status/{job_id}`
Polls background ingestion job status.

**Response states:** `pending` → `reading` → `chunking` → `embedding` → `indexing` → `reloading` → `completed` | `failed`

```json
{
  "status": "completed",
  "message": "Document ingested successfully.",
  "filename": "docs.md",
  "strategy": "markdown",
  "nodes_created": 76,
  "index_name": "AilixirDocs"
}
```

---

#### `GET /rag/status`
RAG system health check.

**Response:**
```json
{
  "weaviate_connected": false,
  "index_name": "AilixirDocs",
  "node_count": -1,
  "engine_ready": true,
  "embed_model": "huggingface/intfloat/multilingual-e5-large-instruct",
  "llm_model": "groq/llama-3.3-70b-versatile",
  "search_mode": "hybrid (α=0.5)",
  "top_k": 5
}
```

---

### 12.2 WebSocket Voice Channel

**Endpoint:** `ws://{host}/ws/voice?session_id={id}`

Full-duplex real-time voice channel. Both sides communicate via JSON messages.

**Client → Server:**

| Message Type | Fields | Description |
|-------------|--------|-------------|
| `audio_chunk` | `data` (base64), `format` | Raw audio chunk during recording |
| `audio_end` | `format` | User finished speaking — triggers transcription |
| `interrupt` | — | Stop current AI streaming response |
| `vad_energy` | `rms` (float) | Energy reading for VAD |

**Server → Client:**

| Message Type | Fields | Description |
|-------------|--------|-------------|
| `vad_status` | `speaking` (bool), `rms` | Voice activity state change |
| `transcript` | `text`, `final` (bool) | Transcription result |
| `ai_token` | `token`, `done` (bool) | Streaming AI response token |
| `ai_done` | — | AI finished responding |
| `interrupted` | — | Confirmed interruption |
| `error` | `message` | Error occurred |

**Interrupt flow:**
- When `audio_chunk` arrives while `session.ai_streaming=True` → sets `session.interrupted=True`
- Streaming loop checks `session.interrupted` each token → breaks early
- Sends `{"type": "interrupted"}` to client

---

# 13. Monitoring System

## 13.1 Overview

The Scientific OS includes a fully integrated in-process monitoring system designed specifically for Hugging Face Spaces deployments where external observability stacks such as Prometheus and Grafana are unavailable.

The monitoring layer collects operational metrics directly inside the FastAPI application and exposes them through JSON APIs and a real-time dashboard.

### Key Features

* Request tracking
* Response latency monitoring
* Agent usage analytics
* Token consumption tracking
* Error logging
* Session monitoring
* Live dashboard
* Zero external dependencies

---

## 13.2 app/monitoring.py

The `monitoring.py` module acts as the central metrics collector.

### Responsibilities

#### Request Metrics

Tracks:

* Total requests
* Endpoint usage
* HTTP status codes
* Request duration

#### Agent Metrics

Tracks execution counts and latency for:

* Chemical Agent
* Medical Agent
* RAG Agent
* APP Agent

#### Token Usage

Stores estimated token consumption per model.

Example:

```python
record_tokens(
    model="llama-3.3-70b",
    prompt_tokens=150,
    completion_tokens=80
)
```

#### Error Tracking

Captures:

* Exceptions
* Agent failures
* External API failures
* RAG retrieval errors

#### Session Analytics

Tracks:

* Active sessions
* Session history
* User activity patterns

---

## 13.3 Dashboard (GET /monitor)

A real-time web dashboard is available at:

```text
GET /monitor
```

### Dashboard Components

#### KPI Cards

Displays:

* Total Requests
* Error Rate
* Active Sessions
* Out-of-Domain Count

#### Latency Analytics

Displays:

* Average Latency
* P50
* P95
* P99
* Minimum
* Maximum

#### Agent Distribution

Visualizes traffic handled by:

* Chemical Agent
* Medical Agent
* RAG Agent
* APP Agent

#### Token Consumption

Shows token usage grouped by model.

#### Error Feed

Displays recent application errors.

#### Request Log

Displays latest requests and response statistics.

### Refresh Strategy

Dashboard automatically refreshes every 10 seconds.

---

## 13.4 Metrics API

### GET /metrics

Returns a full metrics snapshot.

Example:

```json
{
  "requests": 1245,
  "errors": 18,
  "error_rate": 1.4,
  "active_sessions": 23,
  "latency": {
    "avg": 0.82,
    "p95": 1.91,
    "p99": 2.44
  }
}
```

---

### GET /metrics/requests

Returns recent request history.

Example:

```text
GET /metrics/requests?limit=50
```

---

### Middleware Integration

All HTTP traffic is automatically instrumented using FastAPI middleware.

Collected automatically:

* Path
* Method
* Status Code
* Duration
* Timestamp

---

# 14. CI/CD Pipeline

## 14.1 Overview

Deployment is fully automated through GitHub Actions.

Any push to the main branch automatically triggers a deployment to Hugging Face Spaces.

### Workflow

```text
Developer Push
       ↓
GitHub Actions
       ↓
Build Validation
       ↓
Sync Repository
       ↓
Hugging Face Space
       ↓
Automatic Rebuild
       ↓
Production Deployment
```

### Benefits

* Zero manual deployment
* Reproducible builds
* Automatic updates
* Version-controlled infrastructure

---

## 14.2 Workflow File

Location:

```text
.github/workflows/deploy.yml
```

Main responsibilities:

1. Checkout repository
2. Authenticate with Hugging Face
3. Push code to Space repository
4. Trigger Space rebuild

Example:

```yaml
on:
  push:
    branches:
      - main
```

---

## 14.3 GitHub Secret Setup

Required GitHub Secrets:

### HF_TOKEN

Hugging Face Access Token.

### HF_USERNAME

Hugging Face username.

### HF_SPACE_REPO

Target Space repository.

Example:

```text
HF_TOKEN=hf_xxxxxxxxxxxxx
HF_USERNAME=username
HF_SPACE_REPO=username/scientific-os
```

### Secret Location

```text
Repository
 └── Settings
     └── Secrets and Variables
         └── Actions
```

---

## 14.4 Manual Trigger

The deployment workflow can also be executed manually.

### GitHub UI

```text
Actions
 └── Deploy Workflow
     └── Run Workflow
```

### Manual Redeploy Use Cases

* Dependency changes
* Environment variable updates
* Force rebuild
* Recovery after failed deployment

### Verification

After deployment:

```text
HF Space
 └── Build Logs
 └── Runtime Logs
 └── Health Endpoints
```

Recommended health checks:

```text
GET /
GET /health
GET /metrics
GET /monitor
```

Successful responses indicate a healthy deployment and monitoring stack.

---

## 15. Data Persistence on HF Spaces

HF Spaces containers use an **ephemeral filesystem** at `/code`. Everything written there is lost on restart or rebuild. The only persistent path is `/data` — but **only if Persistent Storage is enabled** in Space settings.

| Data | Path | Persistent? | Notes |
|------|------|-------------|-------|
| Vector Index | `/data/rag_index` | ✅ Yes (if /data enabled) | Loaded back on restart via `load_persisted_index()` |
| Long-Term Memory | `/code/app/memory/long_term_store.json` | ❌ **No** | Lost on every restart |
| Embedding Model Cache | HF Hub cache | ✅ Yes (HF caches models) | Re-downloaded on first boot only |
| Short-Term Memory | RAM only | ❌ No | Intentionally ephemeral |
| RQ Job Statuses | RAM dict + Redis | ❌ No (no Redis on HF) | Lost on restart |

**To make Long-Term Memory persistent**, change one line in `app/memory/long_term.py`:
```python
# Before (ephemeral):
self.path = path or os.path.join(os.path.dirname(__file__), "long_term_store.json")

# After (persistent):
self.path = path or "/data/long_term_store.json"
```

**To enable /data persistence:**
1. Go to your HF Space → Settings
2. Persistent Storage → Enable
3. Rebuild the Space

**What happens on restart (without Persistent Storage):**
1. Vector Index (RAM) → lost
2. `/data/rag_index` → also lost (not enabled)
3. System starts with empty knowledge base
4. First `/rag/ingest` call rebuilds the index from scratch

**What happens on restart (with Persistent Storage):**
1. Vector Index (RAM) → lost
2. `/data/rag_index` → survives ✅
3. On startup, `load_persisted_index()` → `StorageContext.from_defaults(persist_dir="/data/rag_index")` → index reloaded into RAM ✅
4. System ready without re-ingestion

---

## 16. Dependency Matrix

All versions pinned to avoid breaking changes between LlamaIndex and Weaviate client:

```
llama-index-core                   >=0.10.1, <0.11.0
llama-index-llms-groq              >=0.1.0,  <0.4.0
llama-index-embeddings-openai      >=0.1.0,  <0.3.0
llama-index-embeddings-huggingface >=0.1.0,  <0.5.0
llama-index-readers-file           >=0.1.0,  <0.4.0
llama-index-vector-stores-weaviate >=1.0.0,  <2.0.0   ← first version supporting weaviate-client v4
weaviate-client                    >=4.5.7,  <5.0.0

fastapi            ==0.104.1
starlette          ==0.27.0
uvicorn            ==0.24.0
httpx              ==0.25.0
pydantic           ==2.5.0
pydantic-settings  ==2.1.0
openai             ==1.6.1
sentence-transformers >=2.2.0
```

**Why these specific versions:**
- `llama-index-vector-stores-weaviate 0.1.x` requires `weaviate-client <4.0` (v3 API) — incompatible with our codebase
- `llama-index-vector-stores-weaviate 1.0.0` was the first release to support `weaviate-client >=4.5.7`
- `llama-index-core <0.11` removed `TextNode.get_doc_id()` → compatibility shim applied in `indexer.py`

---

## 17. Known Bugs & Fixes Applied

All bugs discovered and fixed during development/deployment. The original bug report covered 6 issues:

### Bug 1 — `reload_engine` did not actually reload

**Root cause:** `_initialise()` has an early-return guard:
```python
if self._ready or rag_state["ready"]: return
```
`reload_engine()` set `self._ready = False` but not `rag_state["ready"]`, so `_initialise()` returned immediately without loading the new index.

**Fix:** `app/agents/customer_support/agent.py`
```python
async def reload_engine(self) -> None:
    async with self._lock:
        self._ready = False
        self._query_engine = None
        rag_state["ready"] = False  # ← Added
    await self._initialise()
```

---

### Bug 2 — Index lost on every HF Space restart

**Root cause:** `PERSIST_DIR` pointed to `/code/...` which is ephemeral.

**Fix:** `src/indexer.py`
```python
# Now tries /data first (HF persistent), falls back to local
try:
    _test = _HF_PERSISTENT.parent / ".write_test"
    _test.touch(); _test.unlink()
    PERSIST_DIR = str(_HF_PERSISTENT)      # /data/rag_index
except Exception:
    PERSIST_DIR = str(_LOCAL_FALLBACK)     # .../storage/rag_index
```

---

### Bug 3 — Each ingestion overwrote the entire index

**Root cause:** Every call to `_run_pipeline()` built a brand new `VectorStoreIndex(nodes, ...)`, discarding all previously ingested documents.

**Fix:** `src/ingestion_service.py`
```python
existing = VectorIndexManager._GLOBAL_IN_MEMORY_INDEX
if existing is not None:
    for node in nodes:
        existing.insert(node)   # additive
    index = existing
else:
    index = VectorStoreIndex(nodes, ...)  # first ingestion
```

---

### Bug 4 — Groq has no embeddings API

**Root cause:** Bug report recommended `llama-text-embed-v2` via Groq, but `GET https://api.groq.com/openai/v1/embeddings` returns 404. Groq provides no embeddings endpoint.

**Fix:** `src/embeddings.py`
```python
if provider == "groq":
    logger.warning("Groq has no embeddings API — switching to HuggingFace.")
    provider = "huggingface"
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
```

Switched to `intfloat/multilingual-e5-large-instruct` as the default (SOTA Arabic retrieval, dim=1024).

---

### Bug 5 — Redis timeout wasted 4 seconds on startup

**Root cause:** `socket_connect_timeout=2` and `socket_timeout=2` were set in both `_check_redis_available()` AND `_init_redis()`. On HF Spaces (no Redis), both would time out = 4 seconds wasted before the app started.

**Fix:** `app/main.py`
```python
socket_connect_timeout=0.5,  # 0.5s is sufficient to detect missing Redis
socket_timeout=0.5
```

---

### Bug 6 — `_bootstrap_llama_index` ran on every reload

**Root cause:** `reload_engine()` calls `_initialise()`, which called `_bootstrap_llama_index()` unconditionally. With HuggingFace embeddings, this meant re-downloading the 560MB model on every document ingestion.

**Fix:** `app/agents/customer_support/agent.py`
```python
if not rag_state.get("embedding_initialized") or not rag_state.get("llm_initialized"):
    await loop.run_in_executor(None, self._bootstrap_llama_index)
else:
    logger.info("[RAGAgent] LLM + Embeddings already initialised — skipping bootstrap.")
```

---

### Bug 7 — `'TextNode' object has no attribute 'get_doc_id'`

**Root cause:** `llama-index-core >=0.11` removed `TextNode.get_doc_id()`, but `llama-index-vector-stores-weaviate 1.x` still calls it internally.

**Fix:** `src/indexer.py` — compatibility shim applied at import time:
```python
from llama_index.core.schema import TextNode as _TextNode, BaseNode as _BaseNode
if not hasattr(_TextNode, "get_doc_id"):
    _TextNode.get_doc_id = lambda self: self.ref_doc_id or self.node_id
if not hasattr(_BaseNode, "get_doc_id"):
    _BaseNode.get_doc_id = lambda self: self.ref_doc_id or self.node_id
```

---

### Bug 8 — pip ResolutionImpossible on Docker build

**Root cause:** `llama-index-vector-stores-weaviate 0.1.x` requires `weaviate-client <4.0`, but the codebase uses `weaviate-client >=4.5.7` (v4 API). The version range `>=0.3.0,<0.4.0` doesn't exist on PyPI (package jumped from 0.1.5 to 1.0.0).

**Fix:** `requirements.txt`
```
llama-index-vector-stores-weaviate>=1.0.0,<2.0.0
weaviate-client>=4.5.7,<5.0.0
```

---

## 18. Deployment Guide (HF Spaces)

### Prerequisites

1. Hugging Face account with a Space (Docker SDK)
2. Groq API key from [console.groq.com](https://console.groq.com)

### Step 1: Set Secrets

In your HF Space → Settings → Repository Secrets:

```
GROQ_API_KEY = gsk_xxxxxxxxxxxxxxxxxxxx
```

Optional:
```
OPENAI_API_KEY = sk-xxxx        # For OpenAI TTS
EMBEDDING_PROVIDER = huggingface
EMBEDDING_MODEL = intfloat/multilingual-e5-large-instruct
```

### Step 2: Enable Persistent Storage (Recommended)

Settings → Persistent Storage → Enable

This ensures `/data/rag_index` survives container restarts. Without it, you must re-ingest documents after every restart.

### Step 3: Upload Files

Upload your project files. The Dockerfile handles the build:

```dockerfile
FROM python:3.10-slim
WORKDIR /code
RUN apt-get update && apt-get install -y build-essential
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Step 4: Cold Start Sequence

After first deploy (or after a rebuild):

```
~0s    Container starts
~1s    FastAPI app starts
~1.5s  Redis check fails (0.5s timeout)
~2s    LongTermMemory initialized (JSON fallback)
~2s    RAG _initialise() starts in background
~2s    → Groq LLM initialized (instant)
~30s   → Embedding model download begins
         intfloat/multilingual-e5-large-instruct: ~560MB
         (Cached after first download — subsequent restarts skip this)
~3min  → Embedding model loaded, system READY

First request to /rag/ingest:
~Ns    Document chunked → embedded → persisted to /data/rag_index
         (N depends on document size and number of chunks)
```

> After the embedding model is cached by HF, subsequent cold starts take ~30 seconds instead of ~3 minutes.

### Step 5: Ingest Your Knowledge Base

Use the `/rag/ingest` endpoint to upload Markdown documentation:

```bash
curl -X POST https://your-space.hf.space/rag/ingest \
  -F "file=@your-docs.md" \
  -F "strategy=markdown"
```

Poll for completion:
```bash
curl https://your-space.hf.space/rag/ingest/status/{job_id}
```

Check status:
```bash
curl https://your-space.hf.space/rag/status
```

### Verifying Index Persistence

From the HF Space **Files** tab, after ingesting documents:
- Navigate to `/data/` directory
- You should see `rag_index/` folder containing:
  - `docstore.json`
  - `index_store.json`
  - `vector_store.json`
  - `graph_store.json`

If these files exist, your index will survive restarts.

---

*Documentation generated for Scientific Operating System v1.0 — AI-lixir RAG Pipeline*