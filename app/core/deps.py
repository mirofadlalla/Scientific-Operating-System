"""
app.core.deps
~~~~~~~~~~~~~
Immutable singletons initialised once at import time.

These objects are safe to import directly (no reassignment ever happens):
  from app.core.deps import client, chemical_agent, …
"""
from openai import AsyncOpenAI

from app.config import settings
from app.agents.chemical.agent import ChemicalAgent
from app.agents.medical.agent import MedicalAgent
from app.agents.customer_support.agent import CustomerSupportRAGAgent
from app.orchestrator.brain import OrchestratorBrain
from app.memory.short_term import ShortTermMemory
from app.audio import audio_processor  # noqa: F401  (re-exported for convenience)

# ── OpenAI-compatible client (points at Groq) ─────────────────────────────────
client = AsyncOpenAI(
    base_url=settings.GROQ_BASE_URL,
    api_key=settings.GROQ_API_KEY,
)

# ── Expert domain agents ───────────────────────────────────────────────────────
chemical_agent = ChemicalAgent()
medical_agent  = MedicalAgent()
rag_agent      = CustomerSupportRAGAgent()

# ── Orchestrator + short-term memory ──────────────────────────────────────────
orchestrator = OrchestratorBrain()
short_memory = ShortTermMemory()
