import json
import logging
from openai import OpenAI
from app.config import settings
from app.orchestrator.prompts import ORCHESTRATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class OrchestratorBrain:
    def __init__(self):
        # Configure a client that points to Qwen endpoint when available.
        # Use getattr to avoid AttributeError when settings are not defined.
        self.model = getattr(settings, "QWEN_MODEL", "qwen3-32b")
        qwen_base = getattr(settings, "QWEN_API_BASE", None)
        qwen_key = getattr(settings, "QWEN_API_KEY", None)
        self.client = None
        if qwen_base and qwen_key and qwen_key != "placeholder_key":
            try:
                self.client = OpenAI(base_url=qwen_base, api_key=qwen_key)
            except Exception as e:
                logger.warning(f"Failed to initialize QWen client: {e}")
                self.client = None
        else:
            logger.warning("QWEN API settings not configured; using local fallback classifier.")

    def classify_intent(self, query: str) -> dict:
        """
        Classifies user query into chemical, medical, or app_agent intent using the LLM.
        """
        try:
            # If client is not configured, fall back to local keyword classifier
            if not self.client:
                return self._fallback_classify(query)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            result_text = response.choices[0].message.content
            return json.loads(result_text)
        except Exception as e:
            logger.error(f"Error during intent classification: {e}")
            return self._fallback_classify(query)

    def _fallback_classify(self, query: str) -> dict:
        """Simple local keyword matching router used when the LLM client is unavailable."""
        q = query.lower().strip()

        # Greetings and social messages → always app_agent
        greeting_words = [
            "hello", "hi", "hey", "thanks", "thank", "bye", "goodbye",
            "how are you", "good morning", "good evening", "good night",
            "who are you", "what can you do", "help", "ok", "okay", "yes", "no",
            "مرحبا", "أهلا", "شكرا", "كيف", "سلام", "صباح", "مساء"
        ]
        if any(w in q for w in greeting_words) or len(q.split()) <= 2:
            return {"intent": "app_agent", "reasoning": "Fallback: greeting or short message"}

        # Chemical domain keywords
        chemical_keywords = [
            "smiles", "chemical", "molecule", "atom", "admet", "faiss",
            "compound", "molecular", "drug screen", "similarity", "ligand", "inhibitor"
        ]
        if any(w in q for w in chemical_keywords):
            return {"intent": "chemical", "reasoning": "Fallback match for chemical keywords"}

        # Medical / biomedical keywords
        medical_keywords = [
            "disease", "drug", "clinical", "medical", "symptom", "patient",
            "pathway", "protein", "receptor", "target", "biomarker", "genome",
            "pharmacology", "therapeutic", "enzyme", "gene", "rna", "dna"
        ]
        if any(w in q for w in medical_keywords):
            return {"intent": "medical", "reasoning": "Fallback match for medical keywords"}

        # Default: app_agent
        return {"intent": "app_agent", "reasoning": "Fallback default — no domain keywords matched"}
