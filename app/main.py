import json
import asyncio
import traceback  # Contextual debugging
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI
from fastapi.responses import HTMLResponse
import os

# Import system configuration and custom agents
from app.config import settings
from app.agents.chemical.agent import ChemicalAgent
from app.agents.medical.agent import MedicalAgent
from app.orchestrator.brain import OrchestratorBrain
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory

app = FastAPI(title="AI Scientific OS - Core Router")

client = AsyncOpenAI(
    base_url=settings.GROQ_BASE_URL,
    api_key=settings.GROQ_API_KEY
)

# Initialize expert agents and infrastructure singletons
chemical_agent = ChemicalAgent()
medical_agent = MedicalAgent()

# Orchestrator + Memory
orchestrator = OrchestratorBrain()
short_memory = ShortTermMemory()
long_memory = LongTermMemory()
# Compatibility mapping expected by existing tests and callers
SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

class UserQuery(BaseModel):
    session_id: str
    user_id: str
    text_input: str


def is_general_greeting(text: str) -> bool:
    """
    Checks if the input is a general greeting or casual question that doesn't need agent routing.
    Returns True if the message is just a greeting, casual chat, or general help request.
    """
    text_lower = text.strip().lower()
    
    # General greetings and casual questions (Arabic and English)
    greeting_patterns = [
        r"^(hello|hi|hey|greetings|السلام عليكم|أهلا|مرحبا|كيف حالك|شنو أخبارك|كيفك|كيفك أنت).*$",
        r"^(what is|what's|who are|who is|حكي|شنو|ويش)",  # Vague questions
        r"^(thanks|شكرا|اشكرك|thank you).*$",  # Gratitude
        r"^(bye|goodbye|الوداع|باي|سلام|مع السلامة).*$",  # Goodbye
    ]
    
    for pattern in greeting_patterns:
        if re.match(pattern, text_lower):
            return True
    
    # If message is very short (< 10 chars), likely just greeting
    if len(text_lower.split()) <= 2 and not any(keyword in text_lower for keyword in 
        ["compound", "drug", "disease", "molecule", "chemical", "smiles", "admet", 
         "screening", "pathway", "protein", "target", "مركب", "دواء", "مرض"]):
        return True
    
    return False


def should_skip_orchestrator(text: str) -> bool:
    """
    Returns True if the message can be handled directly without agent routing.
    """
    return is_general_greeting(text)

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
    """
    Main orchestration endpoint for processing user queries.
    Handles both general questions (direct response) and scientific queries (agent routing).
    """
    
    # ============================================================================
    # STEP 0: Quick Check - Is this a general greeting/casual question?
    # ============================================================================
    if should_skip_orchestrator(query.text_input):
        print(f"[DEBUG] Detected general greeting/casual message: skip orchestrator")
        
        async def direct_response_streamer():
            """For casual greetings, provide a direct friendly response"""
            try:
                direct_prompt = f"""
                The user sent a casual message or greeting. Provide a warm, friendly response.
                User message: "{query.text_input}"
                
                Respond in the same language they used (Arabic or English) with a short, warm greeting.
                Keep it brief and friendly.
                """
                
                stream = await client.chat.completions.create(
                    model=settings.ORCHESTRATOR_MODEL,
                    messages=[{"role": "user", "content": direct_prompt}],
                    temperature=0.7,  # More casual, less formal
                    stream=True
                )
                
                full_reply = ""
                async for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        full_reply += token
                        yield token
                
                # Store in memory
                short_memory.add_message(query.session_id, "user", query.text_input)
                short_memory.add_message(query.session_id, "assistant", full_reply)
                SESSION_MEMORY.setdefault(query.session_id, []).append({"role": "user", "content": query.text_input})
                SESSION_MEMORY.setdefault(query.session_id, []).append({"role": "assistant", "content": full_reply})
                long_memory.add_entry(query.session_id, full_reply, metadata={"intent": "GREETING", "agent": "NONE"})
                
            except Exception as e:
                print(f"[STREAM CRASH]: {str(e)}")
                yield f"\n[Stream Error]: {str(e)}"
        
        return StreamingResponse(direct_response_streamer(), media_type="text/plain")
    
    # ============================================================================
    # STEP 1: Normal Flow - Route to appropriate agent
    # ============================================================================
    # If tests or other code populated the compatibility SESSION_MEMORY,
    # sync it into the short-term memory store so the orchestrator sees it.
    if query.session_id in SESSION_MEMORY:
        for m in SESSION_MEMORY.get(query.session_id, [])[:]:
            try:
                short_memory.add_message(query.session_id, m.get("role", "user"), m.get("content", ""))
            except Exception:
                pass

    # Retrieve recent short-term history and consult the orchestrator
    chat_history = short_memory.get_history(query.session_id, limit=6)
    messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}]
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": query.text_input})

    try:
        # 1. Orchestrator inference layer - prefer main client (mocked in tests),
        # fallback to the internal OrchestratorBrain when the async client fails.
        try:
            response = await client.chat.completions.create(
                model=settings.ORCHESTRATOR_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_content = response.choices[0].message.content
            print(f"[DEBUG] Orchestrator Raw JSON: {raw_content}")
            routing_output = json.loads(raw_content)
            target_agent = routing_output.get("target_agent", "APP_AGENT")
            intent = routing_output.get("intent", "UNKNOWN")
            entities = routing_output.get("entities", {})
        except Exception:
            # Fallback to local orchestrator classifier (sync)
            classification = orchestrator.classify_intent(query.text_input)
            intent_raw = (classification.get("intent") or "").lower()
            entities = classification.get("entities") or {}
            if intent_raw == "chemical":
                target_agent = "CHEMICAL_AGENT"
                intent = "CHEMICAL_SIMILARITY"
            elif intent_raw == "medical":
                target_agent = "MEDICAL_AGENT"
                intent = "BIOMEDICAL_MECHANISM"
            else:
                target_agent = "APP_AGENT"
                intent = "APP_HELP"

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
                
                # Persist to short-term and long-term memory only after a successful generation
                short_memory.add_message(query.session_id, "user", query.text_input)
                short_memory.add_message(query.session_id, "assistant", full_reply)
                # Keep compatibility dict in sync for external tests/code
                SESSION_MEMORY.setdefault(query.session_id, []).append({"role": "user", "content": query.text_input})
                SESSION_MEMORY.setdefault(query.session_id, []).append({"role": "assistant", "content": full_reply})
                long_memory.add_entry(query.session_id, full_reply, metadata={"intent": intent, "agent": target_agent})

            except Exception as inner_stream_error:
                print(f"[STREAM CRASH]: {str(inner_stream_error)}")
                yield f"\n[Stream Runtime Error]: Connection disrupted mid-generation: {str(inner_stream_error)}"

        return StreamingResponse(text_streamer(), media_type="text/plain")

    except Exception as e:
        # Clear out current session history on hard crashes to avoid poison-pill loops
        try:
            short_memory.clear(query.session_id)
            SESSION_MEMORY[query.session_id] = []
        except Exception:
            pass
        print(f"[CORE KERNEL CRASH]: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OS Kernel Error Trace: {str(e)}")


# Mount index.html at root directory route
@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()