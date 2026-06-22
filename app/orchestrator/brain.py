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
        # Simple local keyword matching routing fallback logic
        q = query.lower()
        if any(w in q for w in ["smiles", "chemical", "molecule", "atom", "admet", "faiss"]):
            return {"intent": "chemical", "reasoning": "Fallback match for chemical keywords"}
        elif any(w in q for w in ["disease", "drug", "clinical", "medical", "symptom", "patient"]):
            return {"intent": "medical", "reasoning": "Fallback match for medical keywords"}
        else:
            return {"intent": "app_agent", "reasoning": "Fallback default"}
