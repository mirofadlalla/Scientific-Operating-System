import asyncio
from openai import AsyncOpenAI
from app.config import settings, groq_llm_key

class MedicalAgent:
    def __init__(self):
        # Use AsyncOpenAI so we don't block the asyncio event loop
        # Uses GROQ_LLM_API_KEY when set, falls back to GROQ_API_KEY
        self.client = AsyncOpenAI(
            base_url=settings.GROQ_BASE_URL,
            api_key=groq_llm_key()
        )
        self.model_name = getattr(settings, "REASONING_MODEL", settings.ORCHESTRATOR_MODEL)  # openai/gpt-oss-120b

    async def run(self, intent: str, entities: dict, *, user_query: str | None = None) -> str:
        compound = entities.get("compound", "")
        disease  = entities.get("disease", "")
        smiles   = entities.get("smiles", "")

        # System prompt: specialized biomedical reasoning for Drug Discovery
        biomedical_expert_prompt = (
            "You are an advanced AI specializing in Molecular Pharmacology, Drug Discovery, "
            "and Translational Medicine. Your mission is to evaluate biomedical targets, "
            "mechanism of action (MoA), and therapeutic rationale.\n\n"
            "Strict Guidelines:\n"
            "1. Focus on Drug-Target Interactions, receptor bindings, and downstream signaling pathways.\n"
            "2. Evaluate genomic/proteomic feasibility for drug repurposing candidates.\n"
            "3. Use professional, precise biochemical and pharmacodynamics terminology.\n"
            "4. Do NOT output clinical advice, symptoms checklists, or generic health tips."
        )

        if user_query:
            # New path: user's question is the core; entities are supporting context.
            entity_parts = []
            if compound or smiles:
                entity_parts.append(f"  - Compound/SMILES: {smiles if smiles else compound}")
            if disease:
                entity_parts.append(f"  - Disease/Pathology: {disease}")
            entity_context = (
                "\n".join(entity_parts)
                if entity_parts
                else "  (none extracted)"
            )
            user_prompt = (
                f"{user_query}\n\n"
                f"Extracted entities (may be empty):\n{entity_context}"
            )
        elif any(k in intent.lower() for k in ["repurpose", "screen"]):
            user_prompt = (
                f"Analyze the therapeutic rationale for repurposing "
                f"'{compound if compound else smiles}' to treat '{disease}'. "
                "Identify the potential target proteins, downstream cellular pathways, "
                "and biological mechanisms involved."
            )
        else:
            user_prompt = (
                f"Provide a deep pharmacological evaluation and ADMET verification context for:\n"
                f"Chemical Identifier/SMILES: {smiles if smiles else compound}\n"
                f"Target Pathology: {disease}\n"
                "Discuss its structural biology properties, metabolic pathways, and potential efficacy barriers."
            )

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": biomedical_expert_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.0  # Maximum scientific accuracy and consistency
            )
            return f"[Biomedical Reasoning Engine Output]:\n{response.choices[0].message.content}"

        except Exception as e:
            return f"[Medical Agent Error] Failed to generate biomedical validation: {str(e)}"