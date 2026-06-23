import json
import asyncio
import traceback
import re
import math
import struct
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import os

# Import system configuration and custom agents
from app.config import settings
from app.agents.chemical.agent import ChemicalAgent
from app.agents.medical.agent import MedicalAgent
from app.agents.customer_support.agent import CustomerSupportRAGAgent
from app.orchestrator.brain import OrchestratorBrain
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.audio import audio_processor

logger = logging.getLogger(__name__)

# Global variables
worker_process = None
redis_available = False
redis_conn = None
rq_queue = None


def _check_redis_available() -> bool:
    """Check if Redis is available without blocking."""
    try:
        from redis import Redis
        conn = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            socket_connect_timeout=2,
            socket_timeout=2
        )
        conn.ping()
        return True
    except Exception:
        return False


def _init_redis() -> None:
    """Initialize Redis connection and RQ queue if available."""
    global redis_conn, rq_queue, redis_available
    try:
        from redis import Redis
        from rq import Queue
        
        redis_conn = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            socket_connect_timeout=2,
            socket_timeout=2
        )
        redis_conn.ping()
        rq_queue = Queue("default", connection=redis_conn)
        redis_available = True
        logger.info(f"✅ Redis connected: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    except Exception as e:
        redis_available = False
        redis_conn = None
        rq_queue = None
        logger.warning(f"⚠️  Redis connection failed: {e}. Falling back to in-memory mode.")


# ──────────────────────────────────────────────────────────────────────────────
# App Lifespan — startup / shutdown hooks
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-up & teardown hook for long-lived resources."""
    global worker_process, redis_available, long_memory
    
    # Initialize Redis connection (optional)
    _init_redis()
    
    # Initialize long-term memory if Redis is available
    if redis_available:
        try:
            long_memory = LongTermMemory(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB
            )
            logger.info("✅ Long-term memory initialized (Redis-backed)")
        except Exception as e:
            logger.warning(f"Could not initialize long-term memory: {e}")
            long_memory = None
    else:
        logger.info("Long-term memory skipped (Redis not available)")
    
    # 1. Start the RQ worker process only if Redis is available
    if redis_available:
        import subprocess
        import sys
        import os
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd()
            redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
            logger.info(f"Starting RQ worker connecting to Redis: {redis_url}…")
            worker_process = subprocess.Popen(
                [sys.executable, "-m", "rq", "worker", "default", "--url", redis_url],
                env=env,
                cwd=os.getcwd(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info("RQ worker spawned successfully in the background.")
        except Exception as e:
            logger.warning(f"Could not start RQ worker (Redis unavailable): {e}")
    else:
        logger.info("RQ worker skipped (Redis not available)")

    # Startup: pre-initialise the RAG engine in the background
    asyncio.create_task(rag_agent._initialise())
    yield
    # Shutdown: release Weaviate connection and terminate worker
    if worker_process:
        try:
            logger.info("Terminating background RQ worker…")
            worker_process.terminate()
            worker_process.wait(timeout=5)
            logger.info("RQ worker terminated.")
        except Exception as e:
            logger.error(f"Error terminating RQ worker: {e}")

    await rag_agent.close()

app = FastAPI(title="AI Scientific OS — Voice Core", lifespan=lifespan)

client = AsyncOpenAI(
    base_url=settings.GROQ_BASE_URL,
    api_key=settings.GROQ_API_KEY
)

# Initialize expert agents and infrastructure singletons
chemical_agent = ChemicalAgent()
medical_agent  = MedicalAgent()
rag_agent      = CustomerSupportRAGAgent()   # AI-lixir docs RAG agent

# Orchestrator + Memory
orchestrator = OrchestratorBrain()
short_memory = ShortTermMemory()

# Long-term memory requires Redis - initialize after Redis check in lifespan
long_memory: Optional[LongTermMemory] = None

SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

# ──────────────────────────────────────────────────────────────────────────────
# WebSocket connection registry — tracks active voice sessions
# ──────────────────────────────────────────────────────────────────────────────
class VoiceSession:
    """Tracks state for a single WebSocket voice session."""
    def __init__(self, ws: WebSocket, session_id: str):
        self.ws = ws
        self.session_id = session_id
        self.audio_chunks: List[bytes] = []
        self.is_speaking = False          # VAD: user is currently speaking
        self.ai_streaming = False         # AI is currently streaming a response
        self.interrupted = False          # User interrupted AI mid-stream
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = 8        # ~800ms of silence before auto-stop


active_voice_sessions: Dict[str, VoiceSession] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────────────────────
class UserQuery(BaseModel):
    session_id: str
    user_id: str
    text_input: str


class AudioTranscribeRequest(BaseModel):
    audio_format: str = "webm"


class AudioSynthesizeRequest(BaseModel):
    text: str
    voice: str = "nova"


class AudioAgentRequest(BaseModel):
    session_id: str
    user_id: str
    audio_format: str = "webm"
    voice: str = "nova"


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator Helpers
# ──────────────────────────────────────────────────────────────────────────────
def is_general_greeting(text: str) -> bool:
    text_lower = text.strip().lower()
    greeting_patterns = [
        r"^(hello|hi|hey|greetings|السلام عليكم|أهلا|مرحبا|كيف حالك|شنو أخبارك|كيفك).*$",
        r"^(what is|what's|who are|who is|حكي|شنو|ويش)",
        r"^(thanks|شكرا|اشكرك|thank you).*$",
        r"^(bye|goodbye|الوداع|باي|سلام|مع السلامة).*$",
    ]
    for pattern in greeting_patterns:
        if re.match(pattern, text_lower):
            return True
    if len(text_lower.split()) <= 2 and not any(
        kw in text_lower for kw in
        ["compound", "drug", "disease", "molecule", "chemical", "smiles", "admet",
         "screening", "pathway", "protein", "target", "مركب", "دواء", "مرض"]
    ):
        return True
    return False


def should_skip_orchestrator(text: str) -> bool:
    return is_general_greeting(text)


ORCHESTRATOR_SYSTEM_PROMPT = """
You are the central brain (Orchestrator) of an AI Scientific OS specialising in Drug Discovery.
Analyse the user's input, extract entities, and route the query to the correct expert agent.

Available Target Agents and their STRICT Intents:
1. CHEMICAL_AGENT  → CHEMICAL_SIMILARITY | ADMET_ANALYSIS | DRUG_REPURPOSING
   Use for: chemical structures, SMILES, ADMET predictions, molecular similarity, virtual screening.
2. MEDICAL_AGENT   → BIOMEDICAL_MECHANISM
   Use for: biological pathways, drug-target interactions, clinical reasoning, pharmacology.
3. RAG_AGENT       → APP_SUPPORT_RAG
   Use for: questions about AI-lixir platform features, service documentation, API endpoints,
   how-to guides, system overview, ADMET service docs, generation service docs, or any question
   that should be answered from the system documentation knowledge base.
4. APP_AGENT       → APP_HELP
   Use for: casual greetings, general assistant questions, unrelated support.

You MUST respond ONLY with a raw JSON object containing exactly these fields:
{
  "intent": "CHEMICAL_SIMILARITY" | "ADMET_ANALYSIS" | "DRUG_REPURPOSING" | "BIOMEDICAL_MECHANISM" | "APP_SUPPORT_RAG" | "APP_HELP",
  "target_agent": "CHEMICAL_AGENT" | "MEDICAL_AGENT" | "RAG_AGENT" | "APP_AGENT",
  "entities": {"compound": "name if any", "smiles": "SMILES string if any", "disease": "name if any"}
}
"""


async def classify_domain(text_input: str) -> bool:
    """
    Classifies whether the user query is within the domain of the AI Scientific OS.
    Domain includes: Drug Discovery, Chemistry, Biology, Bioinformatics, Cheminformatics,
    Biochemistry, Pharmacology, diseases, target proteins, SMILES, ADMET, and standard app support/greetings.
    General medical queries (like diagnosing a headache, clinical prescriptions), law,
    history, general coding, math, general science, etc. are OUT of domain.
    """
    prompt = f"""
    You are the safety and domain classifier for a Scientific Operating System specializing in Drug Discovery.
    Determine if the following user query is WITHIN the domain or OUT of domain.

    Within-Domain topics:
    - Drug Discovery & Repurposing
    - Chemistry, molecules, compounds, SMILES, ADMET, chemical properties
    - Biology, bioinformatics, proteins, genes, pathways, diseases, target receptors
    - Scientific OS help, features, standard greetings (e.g. hello, hi, how are you, thanks, bye)

    Out-of-Domain topics:
    - General medicine, symptom self-diagnosis, clinical prescriptions, treatment advice, surgery (e.g., "what should I take for headache", "how to cure cancer in humans")
    - Law, legal advice, lawyers, court cases
    - General coding, software engineering (unless related to this app)
    - History, geography, politics, sports, entertainment, general math, general science, cooking, etc.

    User query: "{text_input}"

    Respond ONLY with "IN" if it is within-domain, or "OUT" if it is out-of-domain. Do not add any explanation.
    """
    try:
        response = await client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        verdict = response.choices[0].message.content.strip().upper()
        return "OUT" not in verdict
    except Exception:
        # Fallback to true to avoid blocking on API errors
        return True


async def route_and_stream(text_input: str, session_id: str, user_id: str):
    """
    Shared orchestration logic: routes query through agents and yields
    text tokens from the synthesis stream.
    """
    # Check domain limits
    in_domain = await classify_domain(text_input)
    if not in_domain:
        is_ar = bool(re.search(r"[\u0600-\u06FF]", text_input))
        if is_ar:
            refusal = (
                "عذراً، هذا السؤال خارج نطاق تخصصي العلمي. 🧪\n\n"
                "أنا نظام تشغيل ذكاء اصطناعي علمي متخصص حصرياً في **اكتشاف الأدوية، التحليل الكيميائي، والآليات الطبية الحيوية**. "
                "لا يمكنني الإجابة على الأسئلة المتعلقة بالقوانين، المحاماة، الطب السريري الشخصي، أو أي مجالات عامة أخرى.\n\n"
                "**مجالات تخصصي تشمل:**\n"
                "1. 🧬 **النواة الحيوية**: دراسة المسارات البيولوجية، آليات الأمراض، والبروتينات المستهدفة.\n"
                "2. 🧪 **النواة الكيميائية**: البحث عن المركبات المتشابهة وتوقع الخصائص السمية والحيوية (SMILES & ADMET).\n"
                "3. 🤖 **منسق المهام العلمي**: تشغيل خطوط الفحص الافتراضي وإعادة توجيه الأدوية."
            )
        else:
            refusal = (
                "I'm sorry, this query is outside my scientific domain. 🧪\n\n"
                "I am an AI Scientific OS specializing strictly in **Drug Discovery, Chemical Analysis, and Biomedical Mechanisms**. "
                "I cannot assist with topics like law, clinical medicine, general advice, or other unrelated fields.\n\n"
                "**My core capabilities include:**\n"
                "1. 🧬 **Bioinformatics Core**: Analyzing biological pathways, disease mechanisms, and target receptors.\n"
                "2. 🧪 **Cheminformatics Core**: Searching chemical similarity, predicting ADMET properties, and molecular analysis.\n"
                "3. 🤖 **Scientific Orchestration**: Running virtual screening pipelines for drug repurposing."
            )
        # Yield the tokens of the refusal response dynamically
        for word in refusal.split(" "):
            yield word + " "
            await asyncio.sleep(0.02)
        # Add to memory
        short_memory.add_message(session_id, "user", text_input)
        short_memory.add_message(session_id, "assistant", refusal)
        return
    if should_skip_orchestrator(text_input):
        chat_history = short_memory.get_history(session_id, limit=12)
        messages = [
            {"role": "system", "content": "The user sent a casual message or greeting. Provide a warm, friendly response. Respond in the same language they used (Arabic or English) briefly. Keep the conversation history in context."}
        ]
        for msg in chat_history:
            messages.append(msg)
        messages.append({"role": "user", "content": text_input})

        stream = await client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=messages,
            temperature=0.7,
            stream=True
        )
        full_reply = ""
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_reply += token
                yield token
        short_memory.add_message(session_id, "user", text_input)
        short_memory.add_message(session_id, "assistant", full_reply)
        return

    # Agent routing
    if session_id in SESSION_MEMORY:
        for m in SESSION_MEMORY.get(session_id, [])[:]:
            try:
                short_memory.add_message(session_id, m.get("role", "user"), m.get("content", ""))
            except Exception:
                pass

    chat_history = short_memory.get_history(session_id, limit=12)
    messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}]
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": text_input})

    try:
        response = await client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        raw_content = response.choices[0].message.content
        routing_output = json.loads(raw_content)
        target_agent = routing_output.get("target_agent", "APP_AGENT")
        intent = routing_output.get("intent", "UNKNOWN")
        entities = routing_output.get("entities", {})
    except Exception:
        classification = orchestrator.classify_intent(text_input)
        intent_raw = (classification.get("intent") or "").lower()
        entities = classification.get("entities") or {}
        if intent_raw == "chemical":
            target_agent, intent = "CHEMICAL_AGENT", "CHEMICAL_SIMILARITY"
        elif intent_raw == "medical":
            target_agent, intent = "MEDICAL_AGENT", "BIOMEDICAL_MECHANISM"
        else:
            target_agent, intent = "APP_AGENT", "APP_HELP"

    tasks, task_mapping = [], []
    chemical_intents = ["CHEMICAL_SIMILARITY", "ADMET_ANALYSIS", "DRUG_REPURPOSING"]
    if intent in chemical_intents or target_agent == "CHEMICAL_AGENT":
        tasks.append(chemical_agent.run(intent, entities))
        task_mapping.append("CHEMICAL")
    medical_intents = ["BIOMEDICAL_MECHANISM", "DRUG_REPURPOSING"]
    if entities.get("disease") and (intent in medical_intents or target_agent == "MEDICAL_AGENT"):
        tasks.append(medical_agent.run(intent, entities))
        task_mapping.append("MEDICAL")

    chemical_output = ""
    medical_output  = ""
    rag_output      = ""

    # ── RAG agent for documentation / app-support queries ────────────────────
    if intent == "APP_SUPPORT_RAG" or target_agent == "RAG_AGENT":
        try:
            rag_output = await rag_agent.run(text_input)
        except Exception as exc:
            rag_output = f"[RAG Agent Error]: {exc}"

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                if task_mapping[idx] == "CHEMICAL":
                    chemical_output = f"[Chemical Agent Error]: {res}"
                else:
                    medical_output = f"[Medical Agent Error]: {res}"
            else:
                if task_mapping[idx] == "CHEMICAL":
                    chemical_output = str(res)
                else:
                    medical_output = str(res)

    # ── Build synthesis context ──────────────────────────────────────────────
    if rag_output:
        # RAG answers are already grounded — stream them directly without re-synthesis
        for word in rag_output.split(" "):
            yield word + " "
            await asyncio.sleep(0.008)
        short_memory.add_message(session_id, "user", text_input)
        short_memory.add_message(session_id, "assistant", rag_output)
        SESSION_MEMORY.setdefault(session_id, []).append({"role": "user", "content": text_input})
        SESSION_MEMORY.setdefault(session_id, []).append({"role": "assistant", "content": rag_output})
        long_memory.add_entry(session_id, rag_output, metadata={"intent": intent, "agent": "RAG_AGENT"})
        return

    if chemical_output or medical_output:
        agent_raw_output = f"[Chem Data]: {chemical_output}\n[Bio Data]: {medical_output}".strip()
    else:
        agent_raw_output = "[App System Context]: Standard greeting or help request."

    chat_history = short_memory.get_history(session_id, limit=12)
    messages = [
        {"role": "system", "content": "You are a scientific AI OS assistant. Synthesize a professional, unified answer in the user's conversational language based on the retrieved lab data and the conversation history."}
    ]
    for msg in chat_history:
        messages.append(msg)
    messages.append({
        "role": "user",
        "content": f"User Input Question: \"{text_input}\"\nRetrieved Lab Data: \"{agent_raw_output}\""
    })

    stream = await client.chat.completions.create(
        model=settings.ORCHESTRATOR_MODEL,
        messages=messages,
        temperature=0.3,
        stream=True
    )

    full_reply = ""
    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            full_reply += token
            yield token

    short_memory.add_message(session_id, "user", text_input)
    short_memory.add_message(session_id, "assistant", full_reply)
    SESSION_MEMORY.setdefault(session_id, []).append({"role": "user", "content": text_input})
    SESSION_MEMORY.setdefault(session_id, []).append({"role": "assistant", "content": full_reply})
    long_memory.add_entry(session_id, full_reply, metadata={"intent": intent, "agent": target_agent})


# ──────────────────────────────────────────────────────────────────────────────
# HTTP Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────────────────────────────────────
# RAG Knowledge Base Endpoints & Background Workers
# ──────────────────────────────────────────────────────────────────────────────

class RAGIngestRequest(BaseModel):
    strategy: str = "markdown"   # markdown | sentence | token
    chunk_size: int = 512
    chunk_overlap: int = 20


# Global/Redis job status tracker
ingestion_jobs: Dict[str, Dict[str, Any]] = {}

def update_job_status(job_id: str, status: str, message: str = "", extra_data: dict = None):
    data = {
        "status": status,
        "message": message,
        "filename": (extra_data.get("filename", "") if extra_data else ""),
        "strategy": (extra_data.get("strategy", "") if extra_data else ""),
        "nodes_created": (extra_data.get("nodes_created", 0) if extra_data else 0),
        "index_name": (extra_data.get("index_name", "") if extra_data else ""),
        "error_message": (extra_data.get("error_message", "") if extra_data else "")
    }
    if extra_data:
        data.update(extra_data)

    ingestion_jobs[job_id] = data

    try:
        if long_memory.is_redis:
            long_memory.redis_client.set(f"rag:job:{job_id}", json.dumps(data), ex=3600)  # expires in 1 hour
    except Exception as e:
        logger.error(f"Failed to sync job status to Redis: {e}")

def get_job_status(job_id: str) -> dict:
    try:
        if long_memory.is_redis:
            val = long_memory.redis_client.get(f"rag:job:{job_id}")
            if val:
                return json.loads(val)
    except Exception as e:
        logger.error(f"Failed to fetch job status from Redis: {e}")
    return ingestion_jobs.get(job_id, {"status": "unknown", "message": "Job not found"})


async def run_background_ingest(job_id: str, filename: str, content: bytes, strategy: str):
    def callback(step_name: str, step_msg: str):
        update_job_status(job_id, step_name, step_msg, {"filename": filename, "strategy": strategy})

    try:
        if not rag_agent._ready:
            callback("reading", "Warming up RAG environment...")
            await rag_agent._initialise()

        ingestion_svc = rag_agent.get_ingestion_service()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ingestion_svc.ingest_bytes(
                filename=filename,
                content=content,
                strategy=strategy,
                status_callback=callback
            )
        )

        if result.get("status") == "error":
            error_msg = result.get("message", "Unknown ingestion error.")
            update_job_status(job_id, "failed", f"Ingestion failed: {error_msg}", {
                "filename": filename,
                "strategy": strategy,
                "error_message": error_msg
            })
            return

        update_job_status(job_id, "reloading", "Reloading query engine...", {
            "filename": filename,
            "strategy": strategy,
            "nodes_created": result.get("nodes_created"),
            "index_name": result.get("index_name"),
        })
        await rag_agent.reload_engine()

        update_job_status(job_id, "completed", "Document ingested successfully.", {
            "filename": filename,
            "strategy": strategy,
            "nodes_created": result.get("nodes_created"),
            "index_name": result.get("index_name"),
        })

    except Exception as exc:
        logger.error(f"[run_background_ingest] Unexpected error: {exc}")
        update_job_status(job_id, "failed", f"Ingestion failed: {exc}", {
            "filename": filename,
            "strategy": strategy,
            "error_message": str(exc)
        })


def run_background_ingest_job(job_id: str, filename: str, content: bytes, strategy: str):
    """
    Synchronous wrapper for RQ worker to run the async ingestion pipeline.
    """
    import asyncio
    asyncio.run(run_background_ingest(job_id, filename, content, strategy))


@app.post("/rag/ingest")
async def rag_ingest(
    file: UploadFile = File(...),
    strategy: str    = Form(default="markdown"),
):
    """
    Upload a Markdown (.md) file and ingest it into Weaviate in the background.

    - **file**: The markdown file to ingest.
    - **strategy**: Chunking strategy — 'markdown' (default), 'sentence', or 'token'.

    Returns a job_id for status tracking.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    allowed_ext = {".md", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only {allowed_ext} are accepted."
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        import uuid
        job_id = str(uuid.uuid4())
        
        # Initialize status as pending
        update_job_status(job_id, "pending", "Job scheduled...", {"filename": file.filename, "strategy": strategy})

        # Enqueue background task via RQ
        rq_queue.enqueue(
            run_background_ingest_job,
            job_id,
            file.filename,
            content,
            strategy
        )

        return {
            "status": "success",
            "job_id": job_id,
            "filename": file.filename,
            "strategy": strategy,
            "message": "Ingestion job started in background."
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[/rag/ingest] Unexpected error: {exc}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")


@app.get("/rag/ingest/status/{job_id}")
async def rag_ingest_status(job_id: str):
    """
    Check the status of an active background ingestion job.
    """
    return get_job_status(job_id)


@app.get("/rag/status")
async def rag_status():
    """
    Check the health of the RAG knowledge base.
    Returns Weaviate connectivity, index name, node count, and engine readiness.
    """
    try:
        status = await rag_agent.status()
        return status
    except Exception as exc:
        logger.error(f"[/rag/status] Error: {exc}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {exc}")


@app.post("/orchestrate")
async def process_user_input(query: UserQuery):
    async def streamer():
        try:
            async for token in route_and_stream(query.text_input, query.session_id, query.user_id):
                yield token
        except Exception as e:
            print(f"[STREAM CRASH]: {e}")
            yield f"\n[Stream Error]: {e}"

    return StreamingResponse(streamer(), media_type="text/plain")


# ──────────────────────────────────────────────────────────────────────────────
# Audio HTTP Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/audio/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    audio_format: str = "webm"
):
    """
    Speech-to-Text endpoint using Groq whisper-large-v3-turbo.
    No OpenAI key required — uses existing GROQ_API_KEY.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Detect format from filename if not specified
        if file.filename:
            ext = file.filename.rsplit(".", 1)[-1].lower()
            if ext in {"webm", "mp4", "wav", "mp3", "m4a", "ogg", "flac"}:
                audio_format = ext

        transcribed_text = await audio_processor.transcribe_audio(audio_bytes, audio_format)

        return {
            "status": "success",
            "transcribed_text": transcribed_text,
            "audio_format": audio_format,
            "model": settings.GROQ_WHISPER_MODEL
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.post("/audio/synthesize")
async def synthesize_speech(request: AudioSynthesizeRequest):
    """Text-to-Speech endpoint using OpenAI TTS (requires OPENAI_API_KEY)."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="High-quality TTS requires OPENAI_API_KEY. Use browser SpeechSynthesis instead."
        )
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        audio_bytes = await audio_processor.synthesize_speech(request.text, request.voice)
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {e}")


@app.post("/audio/agent-voice")
async def agent_voice_interaction(
    file: UploadFile = File(...),
    session_id: str = "default_session",
    user_id: str = "default_user",
    audio_format: str = "webm",
    voice: str = "nova"
):
    """HTTP voice-to-voice pipeline (STT → Agent → TTS)."""
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        user_text = await audio_processor.transcribe_audio(audio_bytes, audio_format)
        print(f"[Audio Agent] Transcribed: {user_text}")

        full_response = ""
        async for token in route_and_stream(user_text, session_id, user_id):
            full_response += token

        if not full_response:
            raise HTTPException(status_code=500, detail="Failed to generate response")

        if settings.OPENAI_API_KEY:
            response_audio = await audio_processor.synthesize_speech(full_response, voice)
            return StreamingResponse(
                iter([response_audio]),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=agent_response.mp3",
                    "X-Agent-Text": full_response[:200]
                }
            )
        else:
            return {"status": "text_only", "agent_text": full_response}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice interaction failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Voice Channel  —  /ws/voice
# ──────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/voice")
async def websocket_voice_channel(websocket: WebSocket, session_id: str = "ws_session"):
    """
    Real-time bi-directional voice channel.

    Client → Server messages (JSON):
        {"type": "audio_chunk", "data": "<base64 PCM bytes>", "format": "webm"}
        {"type": "audio_end"}          — user finished speaking
        {"type": "interrupt"}          — interrupt current AI response
        {"type": "vad_energy", "rms": 342.5}  — client-side VAD reading

    Server → Client messages (JSON):
        {"type": "vad_status", "speaking": true/false}
        {"type": "transcript", "text": "...", "final": true/false}
        {"type": "ai_token", "token": "...", "done": false}
        {"type": "ai_done"}
        {"type": "error", "message": "..."}
        {"type": "interrupted"}
    """
    await websocket.accept()
    session = VoiceSession(websocket, session_id)
    active_voice_sessions[session_id] = session
    print(f"[WS] Voice session opened: {session_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ── Audio chunk accumulation ──────────────────────────────────────
            if msg_type == "audio_chunk":
                import base64
                chunk_b64 = msg.get("data", "")
                if chunk_b64:
                    chunk_bytes = base64.b64decode(chunk_b64)
                    session.audio_chunks.append(chunk_bytes)

                # If AI is currently streaming and user sends audio → interrupt
                if session.ai_streaming:
                    session.interrupted = True
                    await websocket.send_text(json.dumps({"type": "interrupted"}))

            # ── VAD energy reading from client ────────────────────────────────
            elif msg_type == "vad_energy":
                rms = float(msg.get("rms", 0))
                speaking = audio_processor.is_speech(rms)
                if speaking != session.is_speaking:
                    session.is_speaking = speaking
                    await websocket.send_text(json.dumps({
                        "type": "vad_status",
                        "speaking": speaking,
                        "rms": rms
                    }))

            # ── User finished speaking — transcribe + respond ─────────────────
            elif msg_type == "audio_end":
                if not session.audio_chunks:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "No audio received"
                    }))
                    continue

                # Transcribe
                audio_format = msg.get("format", "webm")
                try:
                    transcript = await audio_processor.transcribe_chunks(
                        session.audio_chunks, audio_format
                    )
                    session.audio_chunks = []  # reset buffer

                    await websocket.send_text(json.dumps({
                        "type": "transcript",
                        "text": transcript,
                        "final": True
                    }))
                except Exception as exc:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Transcription failed: {exc}"
                    }))
                    session.audio_chunks = []
                    continue

                # Stream AI response
                session.ai_streaming = True
                session.interrupted = False
                try:
                    async for token in route_and_stream(transcript, session_id, "ws_user"):
                        if session.interrupted:
                            break
                        await websocket.send_text(json.dumps({
                            "type": "ai_token",
                            "token": token,
                            "done": False
                        }))
                except Exception as exc:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Agent error: {exc}"
                    }))
                finally:
                    session.ai_streaming = False
                    await websocket.send_text(json.dumps({"type": "ai_done"}))

            # ── Interrupt signal ──────────────────────────────────────────────
            elif msg_type == "interrupt":
                session.interrupted = True
                session.audio_chunks = []
                await websocket.send_text(json.dumps({"type": "interrupted"}))

    except WebSocketDisconnect:
        print(f"[WS] Session disconnected: {session_id}")
    except Exception as exc:
        print(f"[WS] Session error: {exc}")
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        active_voice_sessions.pop(session_id, None)
        print(f"[WS] Session cleaned up: {session_id}")