import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI

# استيراد الإعدادات والـ Agents
from app.config import settings
from app.agents.chemical.agent import ChemicalAgent
from app.agents.medical.agent import MedicalAgent

app = FastAPI(title="AI Scientific OS - Core Router")

client = OpenAI(
    base_url=settings.GROQ_BASE_URL,
    api_key=settings.GROQ_API_KEY
)

# Agents كـ Singletons
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
    final_reply: str #  

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the central brain (Orchestrator) of an AI Scientific OS.
Analyze the user's input, extract entities, and route the query to the correct expert agent.

Available Agents:
1. MEDICAL_AGENT: For clinical questions, drug dosages, pharmacology, and medical RAG.
2. CHEMICAL_AGENT: For molecular similarity, chemical database queries, and ADMET predictions.
3. APP_AGENT: For general settings or app FAQs.

You MUST respond ONLY with a raw JSON object containing exactly these fields:
{
  "intent": "string descriptive intent",
  "target_agent": "MEDICAL_AGENT" | "CHEMICAL_AGENT" | "APP_AGENT",
  "entities": {"compound": "name if any", "disease": "name if any"}
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
        # 1. الـ Orchestrator يحدد الـ Agent والـ Entities
        response = client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0 # صفر عشان الدقة في الـ Routing
        )

        routing_output = json.loads(response.choices[0].message.content)
        target_agent = routing_output.get("target_agent", "APP_AGENT")
        intent = routing_output.get("intent", "unknown")
        entities = routing_output.get("entities", {})

        # 2. توجيه الطلب للـ Agent المختص لتنفيذ الأدوات (Tool Execution Layer)
        agent_raw_output = ""
        if target_agent == "CHEMICAL_AGENT":
            agent_raw_output = await chemical_agent.run(intent, entities)
        elif target_agent == "MEDICAL_AGENT":
            agent_raw_output = await medical_agent.run(intent, entities)
        else:
            agent_raw_output = "[App Agent] مرحباً بك، كيف يمكنني مساعدتك في واجهة التطبيق؟"

        # 3. الـ Response Synthesis (تخلي الـ LLM يصيغ الرد النهائي بناءً على تقرير الـ Agent)
        synthesis_prompt = f"""
        You are the Voice/Text Synthesizer of the Scientific OS.
        The user asked: "{query.text_input}"
        The expert agent returned this raw technical data: "{agent_raw_output}"
        
        Synthesize a natural, professional final response to the user (in the same language they used). 
        Be concise, accurate, and friendly.
        """
        
        synth_response = client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.5
        )
        final_reply = synth_response.choices[0].message.content

        # 4. تحديث الذاكرة
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