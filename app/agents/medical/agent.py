from openai import OpenAI
from app.config import settings

class MedicalAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.GROQ_BASE_URL,
            api_key=settings.GROQ_API_KEY
        )
        self.model_name = settings.ORCHESTRATOR_MODEL # llama-3.3-70b-versatile

    async def run(self, intent: str, entities: dict) -> str:
        compound = entities.get("compound", "")
        disease = entities.get("disease", "")
        smiles = entities.get("smiles", "")

        # الـ System Prompt العلمي المكثف لتقمص دور الـ OpenBioLLM في الـ Drug Discovery
        biomedical_expert_prompt = """
        You are an advanced AI specializing in Molecular Pharmacology, Drug Discovery, and Translational Medicine.
        Your mission is to evaluate biomedical targets, mechanism of action (MoA), and therapeutic rationale.
        
        Strict Guidelines:
        1. Focus deeply on Drug-Target Interactions, receptor bindings, and downstream signaling pathways.
        2. Evaluate genomic/proteomic feasibility for drug repurposing candidates.
        3. Use professional, precise biochemical and pharmacodynamics terminology.
        4. Do NOT output clinical advice, symptoms checklists, or generic health tips.
        """

        if any(k in intent for k in ["repurpose", "screen"]):
            user_prompt = f"""
            Analyze the therapeutic rationale for repurposing '{compound if compound else smiles}' to treat '{disease}'.
            Identify the potential target proteins, downstream cellular pathways, and biological mechanisms involved.
            """
        else:
            user_prompt = f"""
            Provide a deep pharmacological evaluation and ADMET verification context for:
            Chemical Identifier/SMILES: {smiles if smiles else compound}
            Target Pathology: {disease}
            Discuss its structural biology properties, metabolic pathways, and potential efficacy barriers.
            """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": biomedical_expert_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0 # لضمان أعلى دقة علمية وثبات
            )
            return f"[Biomedical Reasoning Engine Output]:\n{response.choices[0].message.content}"

        except Exception as e:
            return f"[Medical Agent Error] Failed to generate biomedical validation: {str(e)}"