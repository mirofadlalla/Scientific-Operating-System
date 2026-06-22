import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI

# Import configuration and agents
from app.config import settings
from app.agents.chemical.agent import ChemicalAgent
from app.agents.medical.agent import MedicalAgent

app = FastAPI(title="AI Scientific OS - Core Router")

client = OpenAI(
    base_url=settings.GROQ_BASE_URL,
    api_key=settings.GROQ_API_KEY
)

# Initialize agents as singletons
chemical_agent = ChemicalAgent()
medical_agent = MedicalAgent()

SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

class UserQuery(BaseModel):
    session_id: str
    user_id: str
    text_input: str

class OrchestratorResponse(BaseModel):
    intent: str
    target_agent: str
    extracted_entities: dict
    agent_raw_output: str
    final_reply: str

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the central brain (Orchestrator) of an AI Scientific OS specializing in Drug Discovery.
Analyze the user's input, extract entities, and route the query to the correct expert agent.

Available Agents:
1. MEDICAL_AGENT: For pharmacological analysis, mechanism of action (MoA), biological pathways, and OpenBioLLM scientific validation.
2. CHEMICAL_AGENT: For virtual screening, molecular similarity, chemical database queries, and ADMET predictions.
3. APP_AGENT: For general settings or app FAQs.

You MUST respond ONLY with a raw JSON object containing exactly these fields:
{
  "intent": "string descriptive intent",
  "target_agent": "MEDICAL_AGENT" | "CHEMICAL_AGENT" | "APP_AGENT",
  "entities": {"compound": "name if any", "smiles": "SMILES string if any", "disease": "name if any"}
}
"""

@app.post("/orchestrate", response_model=OrchestratorResponse)
async def process_user_input(query: UserQuery):
    if query.session_id not in SESSION_MEMORY:
        SESSION_MEMORY[query.session_id] = []

    chat_history = SESSION_MEMORY[query.session_id]
    messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}]

    for msg in chat_history[-6:]:
        messages.append(msg)

    messages.append({"role": "user", "content": query.text_input})

    try:
        # 1. Orchestrator determines the intent, target agent, and extracts entities
        response = client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0  # Set to 0.0 for precise routing accuracy
        )

        routing_output = json.loads(response.choices[0].message.content)
        target_agent = routing_output.get("target_agent", "APP_AGENT")
        intent = routing_output.get("intent", "unknown")
        entities = routing_output.get("entities", {})

        # Dual Routing Pipeline
        chemical_output = ""
        medical_output = ""

        # Trigger Chemical Agent if the intent is chemical/screening related, or if it is the target agent
        if "chemical" in intent or "screen" in intent or target_agent == "CHEMICAL_AGENT":
            chemical_output = await chemical_agent.run(intent, entities)

        # Trigger Medical Agent (OpenBioLLM) if a disease entity is extracted
        if entities.get("disease"):
            medical_output = await medical_agent.run(intent, entities)

        # 2. Consolidate outputs from both agents into a single unified context
        if chemical_output or medical_output:
            agent_raw_output = f"[Chemical Engine Results]:\n{chemical_output}\n\n[Biomedical Reasoning Results]:\n{medical_output}"
        else:
            agent_raw_output = "[App Agent] مرحباً بك، كيف يمكنني مساعدتك في واجهة التطبيق؟"

        # 3. Response Synthesis: Generate the final refined response based on the joint agents report
        synthesis_prompt = f"""
        You are the Voice/Text Synthesizer of the Scientific OS.
        The user asked: "{query.text_input}"
        The expert agents returned this combined raw technical data:
        "{agent_raw_output}"
        
        Synthesize a natural, professional final response to the user (in the same language they used, e.g., Arabic). 
        Integrate both the chemical discovery findings and the biomedical mechanisms seamlessly.
        Be concise, scientifically accurate, and authoritative.
        """
        
        synth_response = client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.3
        )
        final_reply = synth_response.choices[0].message.content

        # 4. Update session memory
        chat_history.append({"role": "user", "content": query.text_input})
        chat_history.append({"role": "assistant", "content": final_reply})

        return OrchestratorResponse(
            intent=intent,
            target_agent=target_agent,
            extracted_entities=entities,
            agent_raw_output=agent_raw_output,
            final_reply=final_reply
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OS Kernel Error: {str(e)}")