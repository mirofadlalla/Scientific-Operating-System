import json
import logging
from openai import OpenAI
from app.config import settings
from app.orchestrator.prompts import ORCHESTRATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class OrchestratorBrain:
    def __init__(self):
        # We configure a client that points to Qwen endpoint or fallback OpenAI
        self.client = OpenAI(
            base_url=settings.QWEN_API_BASE,
            api_key=settings.QWEN_API_KEY
        )
        self.model = "qwen3-32b" # Or equivalent configured Qwen/model name

    def classify_intent(self, query: str) -> dict:
        """
        Classifies user query into chemical, medical, or app_agent intent using the LLM.
        """
        try:
            # Check if API keys are set, otherwise return fallback
            if settings.QWEN_API_KEY == "placeholder_key" or not settings.QWEN_API_KEY:
                logger.warning("QWEN_API_KEY is not set. Returning fallback intent classification.")
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
