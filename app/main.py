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
from app.agents.customer_support.agent import CustomerSupportRAGAgent, rag_state
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
            socket_connect_timeout=0.5,  # Fix Bug 5: 0.5s is enough to detect missing Redis
            socket_timeout=0.5           # avoids 4s wasted on startup in HF Spaces
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
            socket_connect_timeout=0.5,  # Fix Bug 5: fast fail when Redis not present
            socket_timeout=0.5
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

    # ── Always guarantee a working long-term memory ────────────────────────────
    # Try Redis first; fall back to JSON file (works on HF Spaces / any env).
    _init_redis()

    if redis_available:
        try:
            long_memory = LongTermMemory(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB
            )
            logger.info("✅ Long-term memory initialized (Redis-backed)")
        except Exception as e:
            logger.warning(f"Redis long-term memory init failed: {e}. Falling back to JSON file.")
            long_memory = LongTermMemory()  # JSON-file fallback
    else:
        # HF Spaces / local without Redis — use JSON file-backed store
        long_memory = LongTermMemory()
        logger.info("✅ Long-term memory initialized (JSON file-backed — Redis not available)")

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

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse

class ReadinessMiddleware(BaseHTTPMiddleware):
    # Paths that must always respond, even before the RAG engine finishes loading
    _ALWAYS_ALLOW = {"/", "/rag/status", "/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Always let through: UI, health checks, and status endpoints
        if path in self._ALWAYS_ALLOW or path.startswith("/rag/ingest/status"):
            return await call_next(request)

        if not rag_state["ready"]:
            return JSONResponse(
                status_code=503,
                content={"detail": "System is initializing, please wait a few seconds and try again."}
            )
        return await call_next(request)

app = FastAPI(title="AI Scientific OS — Voice Core", lifespan=lifespan)
app.add_middleware(ReadinessMiddleware)

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

# Long-term memory — always initialized in lifespan (JSON fallback guaranteed)
# Set a safe None default here; overwritten in lifespan before first request
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
    """Returns True for greetings, social messages, and casual questions that
    should skip the orchestrator and go straight to the friendly APP_AGENT."""
    text_lower = text.strip().lower()

    # ── Explicit greeting / farewell / gratitude patterns ──────────────────────
    greeting_patterns = [
        # English greetings
        r"^(hello|hi|hey|howdy|greetings|good\s*(morning|afternoon|evening|night|day)).*$",
        r"^(how are you|how('?s| is) it going|how('?s| are) things|what'?s up|sup|yo).*$",
        r"^(nice to meet you|pleased to meet you|good to see you).*$",
        r"^(thanks|thank you|thank you so much|many thanks|cheers|appreciate it).*$",
        r"^(bye|goodbye|see you|take care|later|farewell|have a good one).*$",
        r"^(what can you do|what do you do|who are you|what are you|tell me about yourself).*$",
        r"^(help|i need help|can you help|can you assist).*$",
        r"^(ok|okay|sure|cool|great|awesome|got it|understood|sounds good|perfect|nice).*$",
        r"^(yes|no|maybe|yep|nope|yeah|nah)$",
        r"^(welcome|you'?re welcome|np|no problem|no worries|anytime).*$",
        r"^(sorry|excuse me|my bad|apologies|pardon).*$",
        # Arabic greetings (formal & informal)
        r"^(السلام عليكم|وعليكم السلام|أهلاً|أهلا|مرحباً|مرحبا|هلا|هلو|هاي).*$",
        r"^(كيف حالك|كيف الحال|شلونك|عامل إيه|إيه أخبارك|شنو أخبارك|كيفك|شو أخبارك).*$",
        r"^(صباح الخير|صباح النور|مساء الخير|مساء النور|تصبح على خير).*$",
        r"^(شكراً|شكرا|شكرًا|اشكرك|ممنون|متشكر|جزاك الله خيراً).*$",
        r"^(مع السلامة|باي|وداعاً|في أمان الله|إلى اللقاء|يسلمك).*$",
        r"^(من أنت|ما هو|ماذا تفعل|ماذا تعرف|ما الذي يمكنك|ايش تقدر تسوي).*$",
        r"^(نعم|لا|حسناً|تمام|موافق|صحيح|بالتأكيد|ماشي|اوكي).*$",
        r"^(آسف|عذراً|سامحني|معليش|مع احترامي).*$",
        r"^(سلام)$",
    ]
    for pattern in greeting_patterns:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return True

    # ── Very short messages without any scientific keywords ────────────────────
    scientific_keywords = [
        "compound", "drug", "disease", "molecule", "chemical", "smiles", "admet",
        "screening", "pathway", "protein", "target", "receptor", "ligand", "inhibitor",
        "biomarker", "clinical", "genome", "dna", "rna", "enzyme", "pharmacology",
        "مركب", "دواء", "مرض", "بروتين", "جين", "مسار", "علاج", "دراسة", "تحليل"
    ]
    if len(text_lower.split()) <= 3 and not any(kw in text_lower for kw in scientific_keywords):
        return True

    return False


def should_skip_orchestrator(text: str) -> bool:
    return is_general_greeting(text)


# ── Combined Orchestrator + Domain Classifier prompt ────────────────────────
# This single prompt handles BOTH domain classification AND agent routing,
# eliminating the need for a separate classify_domain() LLM API call.
COMBINED_ORCHESTRATOR_PROMPT = """\
You are the Central Brain of AI-lixir, an AI Scientific Operating System specializing in Drug Discovery.
Your tasks:
  1. Determine if the query is within domain.
  2. If within domain, route it to the correct agent.

Available agents:
  CHEMICAL_AGENT  → intents: CHEMICAL_SIMILARITY | ADMET_ANALYSIS | DRUG_REPURPOSING
      Use for: SMILES, chemical structures, ADMET properties, molecular similarity, virtual screening,
               questions about how ADMET works, MPNN models, CPU/GPU usage in chemistry pipelines,
               any technical question about the chemical analysis system.
  MEDICAL_AGENT   → intent: BIOMEDICAL_MECHANISM
      Use for: biological pathways, drug-target interactions, clinical reasoning, pharmacology,
               disease mechanisms, proteins, receptors, biomarkers, genomics, enzymes.
  RAG_AGENT       → intent: APP_SUPPORT_RAG
      Use for: questions about AI-lixir features, API docs, how-to guides, system documentation,
               "who built this", "who is your master/creator/owner", "what is AI-lixir",
               questions about the platform, its capabilities, or its team.
  APP_AGENT       → intent: APP_HELP
      Use for: greetings, casual chat, short replies, "who are you", "what can you do", thank-yous,
               any ambiguous message that does NOT clearly fit the scientific agents above.

CRITICAL ROUTING RULES:
  - Questions about HOW the system works technically (CPU vs GPU, model architecture, inference speed,
    pipeline design, latency) → CHEMICAL_AGENT or MEDICAL_AGENT depending on context, NOT OUT_OF_DOMAIN.
  - Questions about WHO built the system, WHO owns it, WHO is the creator/master → RAG_AGENT (APP_SUPPORT_RAG).
  - Questions about drugs, molecules, diseases, biology, chemistry → ALWAYS route to scientific agents,
    even if phrased casually or mixed with Arabic.
  - When the topic is REMOTELY related to drug discovery, cheminformatics, or biomedical science → NEVER OUT_OF_DOMAIN.
  - OUT_OF_DOMAIN is ONLY for topics with ZERO connection to science: pure law questions, cooking recipes,
    sports scores, celebrity gossip, weather forecasts, political opinions, financial advice.
  - When in doubt → APP_AGENT. NEVER reject science-adjacent questions.

Respond ONLY with a raw JSON object (no markdown, no explanation):
{
  "intent": "CHEMICAL_SIMILARITY"|"ADMET_ANALYSIS"|"DRUG_REPURPOSING"|"BIOMEDICAL_MECHANISM"|"APP_SUPPORT_RAG"|"APP_HELP"|"OUT_OF_DOMAIN",
  "target_agent": "CHEMICAL_AGENT"|"MEDICAL_AGENT"|"RAG_AGENT"|"APP_AGENT"|"NONE",
  "entities": {"compound": "", "smiles": "", "disease": ""},
  "out_of_domain_reason": "brief reason only when OUT_OF_DOMAIN, else empty string"
}
"""


async def route_and_stream(text_input: str, session_id: str, user_id: str):
    """
    Shared orchestration logic: routes query through agents and yields
    text tokens from the synthesis stream.

    Performance design: greetings are handled with 1 LLM call (streaming response).
    Scientific queries use 1 combined routing call + agent + synthesis = 3 calls total.
    Out-of-domain queries are rejected instantly from the combined routing call output.
    """
    # ── Fast path: greetings skip ALL API calls except the final streamed response ──────
    if should_skip_orchestrator(text_input):
        chat_history = short_memory.get_history(session_id, limit=12)
        messages = [
            {"role": "system", "content": (
                "You are AI-lixir, a friendly and knowledgeable AI Scientific Operating System "
                "specializing in Drug Discovery, Cheminformatics, and Biomedical Research. "
                "You were built by Omar Fadlallah, an AI Engineer and CS student at Mansoura University, Egypt. "
                "The user has sent a casual message, greeting, or conversational input. "
                "Respond warmly and helpfully in the SAME language the user used (Arabic or English). "
                "Be concise and natural. If it's a greeting, introduce yourself briefly and invite them "
                "to ask about drug discovery, molecular analysis, ADMET predictions, or biomedical topics. "
                "If asked who built you, who your master/creator/owner is: Omar Fadlallah. "
                "Never say you cannot help with greetings — always engage positively."
            )}
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
        if long_memory is not None:
            try:
                long_memory.add_entry(session_id, f"User: {text_input}\nAssistant: {full_reply}",
                                      metadata={"intent": "APP_HELP", "agent": "APP_AGENT"})
            except Exception as _lm_err:
                logger.warning(f"long_memory.add_entry failed: {_lm_err}")
        return

    # ── Combined routing + domain classification (single LLM call) ──────────────────
    chat_history = short_memory.get_history(session_id, limit=10)
    messages = [{"role": "system", "content": COMBINED_ORCHESTRATOR_PROMPT}]
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": text_input})

    target_agent = "APP_AGENT"
    intent       = "APP_HELP"
    entities     = {}
    out_of_domain_reason = ""

    try:
        response = await client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        routing_output = json.loads(response.choices[0].message.content)
        intent               = routing_output.get("intent", "APP_HELP")
        target_agent         = routing_output.get("target_agent", "APP_AGENT")
        entities             = routing_output.get("entities") or {}
        out_of_domain_reason = routing_output.get("out_of_domain_reason", "")
    except Exception as exc:
        logger.warning(f"Orchestrator routing failed: {exc}. Using fallback classifier.")
        classification = orchestrator.classify_intent(text_input)
        intent_raw = (classification.get("intent") or "").lower()
        entities   = classification.get("entities") or {}
        if intent_raw == "chemical":
            target_agent, intent = "CHEMICAL_AGENT", "CHEMICAL_SIMILARITY"
        elif intent_raw == "medical":
            target_agent, intent = "MEDICAL_AGENT", "BIOMEDICAL_MECHANISM"
        else:
            target_agent, intent = "APP_AGENT", "APP_HELP"

    # ── Handle out-of-domain inline (no extra API call needed) ───────────────────
    if intent == "OUT_OF_DOMAIN" or target_agent == "NONE":
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
        for word in refusal.split(" "):
            yield word + " "
            await asyncio.sleep(0.02)
        short_memory.add_message(session_id, "user", text_input)
        short_memory.add_message(session_id, "assistant", refusal)
        return

    tasks, task_mapping = [], []
    chemical_intents = ["CHEMICAL_SIMILARITY", "ADMET_ANALYSIS", "DRUG_REPURPOSING"]
    # Only call chemical agent if there's actual molecular data to process
    has_chemical_data = entities.get("smiles") or entities.get("compound")
    if (intent in chemical_intents or target_agent == "CHEMICAL_AGENT") and has_chemical_data:
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
            # If RAG couldn't find info, fall through to APP_AGENT synthesis
            no_info_phrases = [
                "does not contain information",
                "not in the documentation",
                "knowledge base is currently",
                "documentation does not",
                "cannot find",
                "no information"
            ]
            if any(phrase in rag_output.lower() for phrase in no_info_phrases):
                logger.info("[RAG] No relevant docs found — falling back to APP_AGENT synthesis.")
                rag_output = ""
                target_agent = "APP_AGENT"
                intent = "APP_HELP"
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
        if long_memory is not None:
            try:
                long_memory.add_entry(session_id, f"User: {text_input}\nAssistant: {rag_output}",
                                      metadata={"intent": intent, "agent": "RAG_AGENT"})
            except Exception as _lm_err:
                logger.warning(f"long_memory.add_entry failed: {_lm_err}")
        return

    if chemical_output or medical_output:
        agent_raw_output = f"[Chem Data]: {chemical_output}\n[Bio Data]: {medical_output}".strip()
    else:
        agent_raw_output = "[App System Context]: Standard greeting or help request."

    chat_history = short_memory.get_history(session_id, limit=12)
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', text_input))
    lang_instruction = "Respond in Arabic (Egyptian dialect is fine)." if is_arabic else "Respond in English."

    messages = [
        {"role": "system", "content": (
            "You are AI-lixir, a scientific AI OS assistant specializing in Drug Discovery, "
            "Cheminformatics, and Biomedical Research. "
            "You were built by Omar Fadlallah, an AI Engineer from Egypt. "
            "Your job: synthesize a professional, clear answer based on the retrieved lab data and conversation history. "
            f"{lang_instruction} "
            "Use the retrieved data to directly answer the user's question. "
            "If the question is about how the system works technically (CPU, GPU, models, architecture), "
            "explain it clearly based on what you know about the system's design. "
            "NEVER say the question is outside your domain if it relates to science, chemistry, biology, "
            "drug discovery, or how this AI system works. "
            "If asked who built you or who your master/creator is: Omar Fadlallah."
        )}
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
    if long_memory is not None:
        try:
            long_memory.add_entry(
                session_id,
                f"User: {text_input}\nAssistant: {full_reply}",
                metadata={"intent": intent, "agent": target_agent}
            )
        except Exception as _lm_err:
            logger.warning(f"long_memory.add_entry failed: {_lm_err}")


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

    # Sync to Redis if available (graceful fallback if not)
    try:
        if long_memory and hasattr(long_memory, 'is_redis') and long_memory.is_redis:
            long_memory.redis_client.set(f"rag:job:{job_id}", json.dumps(data), ex=3600)  # expires in 1 hour
    except Exception as e:
        logger.warning(f"Could not sync job status to Redis: {e} — Using in-memory storage.")

def get_job_status(job_id: str) -> dict:
    # Try Redis first if available
    try:
        if long_memory and hasattr(long_memory, 'is_redis') and long_memory.is_redis:
            val = long_memory.redis_client.get(f"rag:job:{job_id}")
            if val:
                return json.loads(val)
    except Exception as e:
        logger.warning(f"Could not fetch job status from Redis: {e}")
    # Fall back to in-memory storage
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
    Only used when running via RQ (Redis available).
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # Already in event loop, use create_task instead
        asyncio.create_task(run_background_ingest(job_id, filename, content, strategy))
    except RuntimeError:
        # No running loop, safe to use asyncio.run
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

        # If RQ is available, enqueue background task via Redis
        if rq_queue:
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
                "message": "Ingestion job queued (running in background via Redis)."
            }
        else:
            # Redis unavailable: schedule as async background task
            logger.info(f"Redis unavailable; scheduling ingestion as async task for job {job_id}")
            asyncio.create_task(run_background_ingest(job_id, file.filename, content, strategy))
            
            return {
                "status": "success",
                "job_id": job_id,
                "filename": file.filename,
                "strategy": strategy,
                "message": "Ingestion job scheduled (running in background)."
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