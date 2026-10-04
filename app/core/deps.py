"""
app.core.deps
~~~~~~~~~~~~~
Immutable singletons initialised once at import time.

These objects are safe to import directly (no reassignment ever happens):
  from app.core.deps import client, chemical_agent, …
"""
from openai import AsyncOpenAI

from app.config import settings, groq_llm_key
from app.agents.chemical.agent import ChemicalAgent
from app.agents.medical.agent import MedicalAgent
from app.agents.customer_support.agent import CustomerSupportRAGAgent
from app.orchestrator.brain import OrchestratorBrain
from app.memory.short_term import ShortTermMemory
from app.audio import audio_processor  # noqa: F401  (re-exported for convenience)

# ── OpenAI-compatible client (points at Groq) ─────────────────────────────────
# Uses GROQ_LLM_API_KEY when set, falls back to GROQ_API_KEY
client = AsyncOpenAI(
    base_url=settings.GROQ_BASE_URL,
    api_key=groq_llm_key(),
)


# ── Expert domain agents ───────────────────────────────────────────────────────
chemical_agent = ChemicalAgent()
medical_agent  = MedicalAgent()
rag_agent      = CustomerSupportRAGAgent()

# ── Orchestrator + short-term memory ──────────────────────────────────────────
orchestrator = OrchestratorBrain()
short_memory = ShortTermMemory()
