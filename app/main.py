import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI

# Import system configuration and custom agents
from app.config import settings
from app.agents.chemical.agent import ChemicalAgent
from app.agents.medical.agent import MedicalAgent

app = FastAPI(title="AI Scientific OS - Core Router")

client = OpenAI(
    base_url=settings.GROQ_BASE_URL,
    api_key=settings.GROQ_API_KEY
)

# Initialize expert agents as singletons
chemical_agent = ChemicalAgent()
medical_agent = MedicalAgent()

SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

class UserQuery(BaseModel):
    session_id: str
    user_id: str
    text_input: str

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the central brain (Orchestrator) of an AI Scientific OS specializing in Drug Discovery.
Analyze the user's input, extract entities, and route the query to the correct expert agent using STRICT intents.

Available Target Agents:
1. CHEMICAL_AGENT: For tasks involving chemical structures, screening, and properties.
2. MEDICAL_AGENT: For tasks requiring deep biological pathways or clinical reasoning.
3. APP_AGENT: For general application help.

You MUST choose exactly ONE of the following STRICT Intents:
- "CHEMICAL_SIMILARITY": Search for similar compounds using FAISS/SMILES.
- "ADMET_ANALYSIS": Predict absorption, distribution, metabolism, excretion, and toxicity.
- "DRUG_REPURPOSING": Virtual screening pipeline for a disease to find drug candidates.
- "BIOMEDICAL_MECHANISM": Deep dive into biological pathways, targets, and mechanisms of action.
- "APP_HELP": General application support.

You MUST respond ONLY with a raw JSON object containing exactly these fields:
{
  "intent": "CHEMICAL_SIMILARITY" | "ADMET_ANALYSIS" | "DRUG_REPURPOSING" | "BIOMEDICAL_MECHANISM" | "APP_HELP",
  "target_agent": "CHEMICAL_AGENT" | "MEDICAL_AGENT" | "APP_AGENT",
  "entities": {"compound": "name if any", "smiles": "SMILES string if any", "disease": "name if any"}
}
"""

@app.post("/orchestrate")
async def process_user_input(query: UserQuery):
    if query.session_id not in SESSION_MEMORY:
        SESSION_MEMORY[query.session_id] = []

    chat_history = SESSION_MEMORY[query.session_id]
    messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}]

    for msg in chat_history[-6:]:
        messages.append(msg)

    messages.append({"role": "user", "content": query.text_input})

    try:
        # 1. Orchestrator classifies the intent, routes targets, and extracts entities via JSON object
        response = client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0 
        )

        routing_output = json.loads(response.choices[0].message.content)
        target_agent = routing_output.get("target_agent", "APP_AGENT")
        intent = routing_output.get("intent", "UNKNOWN")
        entities = routing_output.get("entities", {})

        # 2. Execute targeted expert agents in parallel using asyncio tasks
        tasks = []
        task_mapping = []

        chemical_intents = ["CHEMICAL_SIMILARITY", "ADMET_ANALYSIS", "DRUG_REPURPOSING"]
        if intent in chemical_intents or target_agent == "CHEMICAL_AGENT":
            tasks.append(chemical_agent.run(intent, entities))
            task_mapping.append("CHEMICAL")

        medical_intents = ["BIOMEDICAL_MECHANISM", "DRUG_REPURPOSING"]
        if entities.get("disease") and (intent in medical_intents or target_agent == "MEDICAL_AGENT"):
            tasks.append(medical_agent.run(intent, entities))
            task_mapping.append("MEDICAL")

        chemical_output = ""
        medical_output = ""
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, res in enumerate(results):
                if not isinstance(res, Exception):
                    if task_mapping[idx] == "CHEMICAL": 
                        chemical_output = res
                    else: 
                        medical_output = res

        # Assemble tightly payload-trimmed inputs to minimize context token usage
        if chemical_output or medical_output:
            agent_raw_output = f"[Chem]: {chemical_output}\n[Bio]: {medical_output}".strip()
        else:
            agent_raw_output = "[App]: Welcome to Scientific OS."

        # 3. Streaming Engine Layer
        # Generator function to stream tokens instantly and update chat state asynchronously
        async def text_streamer():
            synthesis_prompt = f"""
            User: "{query.text_input}"
            Data: "{agent_raw_output}"
            Synthesize a concise, expert final response in the user's language (e.g., Arabic). Blend findings smoothly.
            """
            
            # Non-blocking async chunk streaming setup by executing synchronous SDK execution inside a thread pool
            def get_openai_stream():
                return client.chat.completions.create(
                    model=settings.ORCHESTRATOR_MODEL,
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    temperature=0.3,
                    stream=True
                )
            
            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(None, get_openai_stream)
            
            full_reply = ""
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_reply += token
                    yield token  # Push the generated token immediately to the client
                    await asyncio.sleep(0.01)  # Yield execution back to the loop to support high concurrency streaming

            # Append complete round-trip logs to the session history in background memory once the stream ends
            chat_history.append({"role": "user", "content": query.text_input})
            chat_history.append({"role": "assistant", "content": full_reply})

        return StreamingResponse(text_streamer(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OS Kernel Error: {str(e)}")
