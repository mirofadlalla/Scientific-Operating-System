# Hugging Face Spaces Deployment Guide

## Overview

Your Scientific OS app now runs on **Hugging Face Spaces** without requiring external services (Redis, Weaviate, Docker).

### What Works on HF Spaces

✅ Voice agents (Chemical, Medical, Customer Support)  
✅ Orchestrator brain  
✅ LLM (Groq) integration  
✅ RAG ingestion (background tasks)  
✅ In-memory storage (session-based)  

### Limitations on HF Spaces

⚠️ No persistent database (Weaviate)  
⚠️ No message queue (Redis)  
⚠️ No long-term memory across restarts  
⚠️ Single-instance only (can't scale horizontally)  

---

## Deployment to HF Spaces

### 1. Create a Space

Go to https://huggingface.co/spaces and create a new Space:
- **Space name**: `scientific-os` (or your preference)
- **Space type**: `Docker`
- **License**: `MIT` (or your choice)

### 2. Push Your Code

```bash
# Initialize git repo (if not already)
git init

# Add HF Spaces as remote
git remote add space https://huggingface.co/spaces/YourUsername/scientific-os

# Push code
git add .
git commit -m "Initial deployment to HF Spaces"
git push space main
```

### 3. Set Secrets

In your Space settings, add **Secrets**:

```
GROQ_API_KEY = your_groq_api_key_here
```

### 4. Configure Space Settings

- **Space type**: Docker
- **CPU**: Medium (recommended) or better for faster ingestion
- **GPU**: Optional (not needed)
- **Persistent storage**: Not needed (no Weaviate to persist)

---

## How Ingestion Works on HF Spaces

### Without Weaviate

On HF Spaces, document ingestion is **scheduled as a background task** but **not persisted**:

```python
# Upload a document
POST /rag/ingest
  file: guide.md
  strategy: markdown

# Returns immediately
{
  "status": "success",
  "job_id": "uuid-string",
  "message": "Ingestion job scheduled (running in background)."
}

# Check status
GET /rag/ingest/status/{job_id}
# Returns: {"status": "reading", ...}
# Then: {"status": "completed", "nodes_created": 42}
```

### What Happens

1. File is uploaded
2. Task scheduled in background (non-blocking)
3. Documents are chunked and embedded
4. Embeddings **stored in memory** (not persistent)
5. Query engine builds from in-memory data
6. When Space restarts, all data is lost

### For Persistent Storage

To add persistent storage on HF Spaces:

1. **Request persistent storage** in Space settings
2. Save documents to `/data/documents/`
3. Modify ingestion to load from disk on startup

---

## Testing Your Deployment

### 1. Check if App Is Running

```bash
curl https://yourspace.hf.space/
# Should return HTML
```

### 2. Check RAG Status

```bash
curl https://yourspace.hf.space/rag/status
# Should return JSON with engine status
```

### 3. Test Ingestion

```bash
# Create a test markdown file
cat > test.md << 'EOF'
# Python Basics

Python is a programming language.

## Variables
Variables store data values.
EOF

# Upload it
curl -X POST https://yourspace.hf.space/rag/ingest \
  -F "file=@test.md" \
  -F "strategy=markdown"

# Should return job_id - remember it

# Check status (replace JID with actual ID)
curl https://yourspace.hf.space/rag/ingest/status/JID
# Poll until status is "completed"
```

### 4. Test Query

```bash
curl https://yourspace.hf.space/rag/query \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

---

## Key Implementation Details

### Background Tasks (No Redis)

When Redis is unavailable (as on HF Spaces):

```python
# Instead of:
rq_queue.enqueue(ingest_job, ...)  # Requires Redis

# We do:
asyncio.create_task(ingest_job(...))  # Uses event loop
```

This allows ingestion to run without blocking the API.

### Job Status Storage

- **With Redis**: Persisted in Redis (survives restarts)
- **Without Redis**: Stored in Python dict (lost on restart)

```python
ingestion_jobs = {}  # In-memory storage

def update_job_status(job_id, status, message, extra):
    ingestion_jobs[job_id] = {...}
```

### LLM & Embeddings

Both use **Groq's API** (external service):
- LLM: `llama-3.3-70b-versatile`
- Embeddings: `text-embedding-3-small`

Requires `GROQ_API_KEY` environment variable.

---

## Troubleshooting

### Space Won't Start

Check logs by clicking "View Logs" in your Space:

```
ERROR: GROQ_API_KEY not set
```

**Fix**: Add secret in Space settings

```
ERROR: Cannot import 'redis'
```

**Fix**: Already handled — app skips Redis if unavailable

### Ingestion Takes Too Long

HF Spaces CPU is shared. Optimization:
- Reduce chunk size: `?chunk_size=256`
- Use faster strategy: `strategy=token`
- Smaller files first (test with 1-2KB markdown)

### Memory Issues

Backend is **in-memory only**, so large documents consume Space RAM:
- Recommended: Keep documents < 10MB total
- Split large files into multiple uploads
- Space restarts periodic clear memory

### RAG Queries Return Empty

**Expected behavior on HF Spaces**:
1. First deployment = no documents ingested
2. Upload documents first
3. Wait for ingestion to complete
4. Then query

---

## Optimization Tips

### 1. Cache Expensive Operations

```python
# Don't re-download embedding model
@lru_cache(maxsize=1)
def get_embedding_model():
    return OpenAIEmbedding(model="text-embedding-3-small", ...)
```

### 2. Batch Small Files

Upload multiple markdown files in a single request:
```python
# Better: Single request, multiple docs
for doc in docs:
    upload_to_rag(doc)
```

### 3. Monitor Space Usage

HF Spaces shows:
- Memory usage (target: 80% of allocated)
- CPU usage (target: consistent < 100%)
- Restart history (should be rare)

---

## Production Considerations

### If You Need Persistence

Upgrade from HF Spaces to Production:

1. **Docker Compose** (local)
   - Full Redis + Weaviate
   - See DEPLOYMENT_GUIDE.md

2. **Cloud Deployment** (AWS/GCP/Azure)
   - Use managed services (Redis via ElastiCache, Weaviate via self-hosted)
   - See main deployment guide

3. **Self-Hosted Kubernetes**
   - Full control, high complexity
   - Consider using Helm charts for Weaviate

### Migration Path

```
HF Spaces (free, limited)
       ↓
Docker Compose (local/dev)
       ↓
Cloud (production)
```

---

## Example: Complete Ingestion Flow

```bash
# 1. Upload document
curl -X POST \
  https://yourspace.hf.space/rag/ingest \
  -F "file=@chemistry.md" \
  -F "strategy=markdown"

# Response:
# {
#   "status": "success",
#   "job_id": "abc-123-def",
#   "message": "Ingestion job scheduled..."
# }

# 2. Poll status
curl https://yourspace.hf.space/rag/ingest/status/abc-123-def

# Response (immediately):
# {"status": "pending", "message": "Job scheduled..."}

# Response (after 5 seconds):
# {"status": "reading", "message": "Reading file..."}

# Response (after 10 seconds):
# {"status": "chunking", "message": "Chunking document..."}

# Response (after 20 seconds):
# {"status": "embedding", "message": "Generating embeddings..."}

# Response (after 30 seconds):
# {
#   "status": "completed",
#   "message": "Document ingested successfully.",
#   "nodes_created": 47,
#   "index_name": "AilixirDocs"
# }

# 3. Query the knowledge base
curl -X POST \
  https://yourspace.hf.space/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is pH equilibrium?"}'

# Response:
# {
#   "answer": "pH equilibrium refers to the...",
#   "status": "success"
# }
```

---

## Support

Having issues? Check:

1. **Logs** → Space "View Logs" button
2. **Status endpoint** → `/rag/status`
3. **Job status** → `/rag/ingest/status/{job_id}`

If stuck:
- Share the full error message from logs
- Mention the exact endpoint you're calling
- Include request/response bodies
