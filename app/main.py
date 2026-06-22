import json
import asyncio
import traceback  # Contextual debugging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import AsyncOpenAI
from fastapi.responses import HTMLResponse
import os

# Import system configuration and custom agents
from app.config import settings
from app.agents.chemical.agent import ChemicalAgent
from app.agents.medical.agent import MedicalAgent

app = FastAPI(title="AI Scientific OS - Core Router")

client = AsyncOpenAI(
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
    # Wipe corrupted memory states if any server crash loops happen
    if query.session_id not in SESSION_MEMORY:
        SESSION_MEMORY[query.session_id] = []

    chat_history = SESSION_MEMORY[query.session_id]
    messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}]

    # Build reliable context history
    for msg in chat_history[-6:]:
        messages.append(msg)

    messages.append({"role": "user", "content": query.text_input})

    try:
        # 1. Orchestrator inference layer
        response = await client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0 
        )

        raw_content = response.choices[0].message.content
        print(f"[DEBUG] Orchestrator Raw JSON: {raw_content}") # Print to terminal for visibility
        
        routing_output = json.loads(raw_content)
        target_agent = routing_output.get("target_agent", "APP_AGENT")
        intent = routing_output.get("intent", "UNKNOWN")
        entities = routing_output.get("entities", {})

        # 2. Parallel agent execution logic
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
            print(f"[DEBUG] Dispatching {len(tasks)} parallel agent tasks...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for idx, res in enumerate(results):
                if isinstance(res, Exception):
                    print(f"[AGENT CRASH] {task_mapping[idx]} Agent failed: {str(res)}")
                    traceback.print_tb(res.__traceback__)
                    # Catch the inner agent failure instead of killing the core OS kernel loop
                    if task_mapping[idx] == "CHEMICAL":
                        chemical_output = f"[Chemical Agent Exception Fail]: {str(res)}"
                    else:
                        medical_output = f"[Medical Agent Exception Fail]: {str(res)}"
                else:
                    if task_mapping[idx] == "CHEMICAL": 
                        chemical_output = str(res)
                    else: 
                        medical_output = str(res)

        # Build clean payloads for synthesis execution
        if chemical_output or medical_output:
            agent_raw_output = f"[Chem Data]: {chemical_output}\n[Bio Data]: {medical_output}".strip()
        else:
            agent_raw_output = "[App System Context]: Standard greeting or help request processed."

        print(f"[DEBUG] Combined Agent Output Length: {len(agent_raw_output)} chars")

        # 3. Secure Async Streaming Engine
        async def text_streamer():
            try:
                synthesis_prompt = f"""
                User Input Question: "{query.text_input}"
                Retrieved Lab Data: "{agent_raw_output}"
                
                Synthesize an expert, polished response in the user's conversational language (e.g., Arabic). 
                Provide a unified, highly professional explanation.
                """
                
                stream = await client.chat.completions.create(
                    model=settings.ORCHESTRATOR_MODEL,
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    temperature=0.3,
                    stream=True
                )
                
                full_reply = ""
                async for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        full_reply += token
                        yield token
                
                # Update chat history state only after a flawless full-stream cycle completion
                chat_history.append({"role": "user", "content": query.text_input})
                chat_history.append({"role": "assistant", "content": full_reply})

            except Exception as inner_stream_error:
                print(f"[STREAM CRASH]: {str(inner_stream_error)}")
                yield f"\n[Stream Runtime Error]: Connection disrupted mid-generation: {str(inner_stream_error)}"

        return StreamingResponse(text_streamer(), media_type="text/plain")

    except Exception as e:
        # Clear out current session history on hard crashes to avoid poison-pill loops
        if query.session_id in SESSION_MEMORY:
            SESSION_MEMORY[query.session_id] = []
        print(f"[CORE KERNEL CRASH]: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OS Kernel Error Trace: {str(e)}")


# Mount index.html at root directory route
@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()