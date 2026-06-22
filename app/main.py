import json
import asyncio
import traceback
import re
import math
import struct
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import os

# Import system configuration and custom agents
from app.config import settings
from app.agents.chemical.agent import ChemicalAgent
from app.agents.medical.agent import MedicalAgent
from app.orchestrator.brain import OrchestratorBrain
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.audio import audio_processor

app = FastAPI(title="AI Scientific OS — Voice Core")

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
SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

# ──────────────────────────────────────────────────────────────────────────────
# WebSocket connection registry — tracks active voice sessions
# ──────────────────────────────────────────────────────────────────────────────
class VoiceSession:
    """Tracks state for a single WebSocket voice session."""
    def __init__(self, ws: WebSocket, session_id: str):
        self.ws = ws
        self.session_id = session_id
        self.audio_chunks: List[bytes] = []
        self.is_speaking = False          # VAD: user is currently speaking
        self.ai_streaming = False         # AI is currently streaming a response
        self.interrupted = False          # User interrupted AI mid-stream
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = 8        # ~800ms of silence before auto-stop


active_voice_sessions: Dict[str, VoiceSession] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────────────────────
class UserQuery(BaseModel):
    session_id: str
    user_id: str
    text_input: str


class AudioTranscribeRequest(BaseModel):
    audio_format: str = "webm"


class AudioSynthesizeRequest(BaseModel):
    text: str
    voice: str = "nova"


class AudioAgentRequest(BaseModel):
    session_id: str
    user_id: str
    audio_format: str = "webm"
    voice: str = "nova"


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator Helpers
# ──────────────────────────────────────────────────────────────────────────────
def is_general_greeting(text: str) -> bool:
    text_lower = text.strip().lower()
    greeting_patterns = [
        r"^(hello|hi|hey|greetings|السلام عليكم|أهلا|مرحبا|كيف حالك|شنو أخبارك|كيفك).*$",
        r"^(what is|what's|who are|who is|حكي|شنو|ويش)",
        r"^(thanks|شكرا|اشكرك|thank you).*$",
        r"^(bye|goodbye|الوداع|باي|سلام|مع السلامة).*$",
    ]
    for pattern in greeting_patterns:
        if re.match(pattern, text_lower):
            return True
    if len(text_lower.split()) <= 2 and not any(
        kw in text_lower for kw in
        ["compound", "drug", "disease", "molecule", "chemical", "smiles", "admet",
         "screening", "pathway", "protein", "target", "مركب", "دواء", "مرض"]
    ):
        return True
    return False


def should_skip_orchestrator(text: str) -> bool:
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


async def classify_domain(text_input: str) -> bool:
    """
    Classifies whether the user query is within the domain of the AI Scientific OS.
    Domain includes: Drug Discovery, Chemistry, Biology, Bioinformatics, Cheminformatics,
    Biochemistry, Pharmacology, diseases, target proteins, SMILES, ADMET, and standard app support/greetings.
    General medical queries (like diagnosing a headache, clinical prescriptions), law,
    history, general coding, math, general science, etc. are OUT of domain.
    """
    prompt = f"""
    You are the safety and domain classifier for a Scientific Operating System specializing in Drug Discovery.
    Determine if the following user query is WITHIN the domain or OUT of domain.

    Within-Domain topics:
    - Drug Discovery & Repurposing
    - Chemistry, molecules, compounds, SMILES, ADMET, chemical properties
    - Biology, bioinformatics, proteins, genes, pathways, diseases, target receptors
    - Scientific OS help, features, standard greetings (e.g. hello, hi, how are you, thanks, bye)

    Out-of-Domain topics:
    - General medicine, symptom self-diagnosis, clinical prescriptions, treatment advice, surgery (e.g., "what should I take for headache", "how to cure cancer in humans")
    - Law, legal advice, lawyers, court cases
    - General coding, software engineering (unless related to this app)
    - History, geography, politics, sports, entertainment, general math, general science, cooking, etc.

    User query: "{text_input}"

    Respond ONLY with "IN" if it is within-domain, or "OUT" if it is out-of-domain. Do not add any explanation.
    """
    try:
        response = await client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        verdict = response.choices[0].message.content.strip().upper()
        return "OUT" not in verdict
    except Exception:
        # Fallback to true to avoid blocking on API errors
        return True


async def route_and_stream(text_input: str, session_id: str, user_id: str):
    """
    Shared orchestration logic: routes query through agents and yields
    text tokens from the synthesis stream.
    """
    # Check domain limits
    in_domain = await classify_domain(text_input)
    if not in_domain:
        is_ar = bool(re.search(r"[\u0600-\u06FF]", text_input))
        if is_ar:
            refusal = (
                "عذراً، هذا السؤال خارج نطاق تخصصي العلمي. 🧪\n\n"
                "أنا نظام تشغيل ذكاء اصطناعي علمي متخصص حصرياً في **اكتشاف الأدوية، التحليل الكيميائي، والآليات الطبية الحيوية**. "
                "لا يمكنني الإجابة على الأسئلة المتعلقة بالقوانين، المحاماة، الطب السريري الشخصي، أو أي مجالات عامة أخرى.\n\n"
                "**مجالات تخصصي تشمل:**\n"
                "1. 🧬 **النواة الحيوية**: دراسة المسارات البيولوجية، آليات الأمراض، والبروتينات المستهدفة.\n"
                "2. 🧪 **النواة الكيميائية**: البحث عن المركبات المتشابهة وتوقع الخصائص السمية والحيوية (SMILES & ADMET).\n"
                "3. 🤖 **منسق المهام العلمي**: تشغيل خطوط الفحص الافتراضي وإعادة توجيه الأدوية."
            )
        else:
            refusal = (
                "I'm sorry, this query is outside my scientific domain. 🧪\n\n"
                "I am an AI Scientific OS specializing strictly in **Drug Discovery, Chemical Analysis, and Biomedical Mechanisms**. "
                "I cannot assist with topics like law, clinical medicine, general advice, or other unrelated fields.\n\n"
                "**My core capabilities include:**\n"
                "1. 🧬 **Bioinformatics Core**: Analyzing biological pathways, disease mechanisms, and target receptors.\n"
                "2. 🧪 **Cheminformatics Core**: Searching chemical similarity, predicting ADMET properties, and molecular analysis.\n"
                "3. 🤖 **Scientific Orchestration**: Running virtual screening pipelines for drug repurposing."
            )
        # Yield the tokens of the refusal response dynamically
        for word in refusal.split(" "):
            yield word + " "
            await asyncio.sleep(0.02)
        # Add to memory
        short_memory.add_message(session_id, "user", text_input)
        short_memory.add_message(session_id, "assistant", refusal)
        return
    if should_skip_orchestrator(text_input):
        direct_prompt = f"""
        The user sent a casual message or greeting. Provide a warm, friendly response.
        User message: "{text_input}"
        Respond in the same language they used (Arabic or English) briefly.
        """
        stream = await client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=[{"role": "user", "content": direct_prompt}],
            temperature=0.7,
            stream=True
        )
        full_reply = ""
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_reply += token
                yield token
        short_memory.add_message(session_id, "user", text_input)
        short_memory.add_message(session_id, "assistant", full_reply)
        return

    # Agent routing
    if session_id in SESSION_MEMORY:
        for m in SESSION_MEMORY.get(session_id, [])[:]:
            try:
                short_memory.add_message(session_id, m.get("role", "user"), m.get("content", ""))
            except Exception:
                pass

    chat_history = short_memory.get_history(session_id, limit=6)
    messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}]
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": text_input})

    try:
        response = await client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        raw_content = response.choices[0].message.content
        routing_output = json.loads(raw_content)
        target_agent = routing_output.get("target_agent", "APP_AGENT")
        intent = routing_output.get("intent", "UNKNOWN")
        entities = routing_output.get("entities", {})
    except Exception:
        classification = orchestrator.classify_intent(text_input)
        intent_raw = (classification.get("intent") or "").lower()
        entities = classification.get("entities") or {}
        if intent_raw == "chemical":
            target_agent, intent = "CHEMICAL_AGENT", "CHEMICAL_SIMILARITY"
        elif intent_raw == "medical":
            target_agent, intent = "MEDICAL_AGENT", "BIOMEDICAL_MECHANISM"
        else:
            target_agent, intent = "APP_AGENT", "APP_HELP"

    tasks, task_mapping = [], []
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
            if isinstance(res, Exception):
                if task_mapping[idx] == "CHEMICAL":
                    chemical_output = f"[Chemical Agent Error]: {res}"
                else:
                    medical_output = f"[Medical Agent Error]: {res}"
            else:
                if task_mapping[idx] == "CHEMICAL":
                    chemical_output = str(res)
                else:
                    medical_output = str(res)

    if chemical_output or medical_output:
        agent_raw_output = f"[Chem Data]: {chemical_output}\n[Bio Data]: {medical_output}".strip()
    else:
        agent_raw_output = "[App System Context]: Standard greeting or help request."

    synthesis_prompt = f"""
    User Input Question: "{text_input}"
    Retrieved Lab Data: "{agent_raw_output}"

    Synthesize an expert, polished response in the user's conversational language.
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

    short_memory.add_message(session_id, "user", text_input)
    short_memory.add_message(session_id, "assistant", full_reply)
    SESSION_MEMORY.setdefault(session_id, []).append({"role": "user", "content": text_input})
    SESSION_MEMORY.setdefault(session_id, []).append({"role": "assistant", "content": full_reply})
    long_memory.add_entry(session_id, full_reply, metadata={"intent": intent, "agent": target_agent})


# ──────────────────────────────────────────────────────────────────────────────
# HTTP Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/orchestrate")
async def process_user_input(query: UserQuery):
    async def streamer():
        try:
            async for token in route_and_stream(query.text_input, query.session_id, query.user_id):
                yield token
        except Exception as e:
            print(f"[STREAM CRASH]: {e}")
            yield f"\n[Stream Error]: {e}"

    return StreamingResponse(streamer(), media_type="text/plain")


# ──────────────────────────────────────────────────────────────────────────────
# Audio HTTP Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/audio/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    audio_format: str = "webm"
):
    """
    Speech-to-Text endpoint using Groq whisper-large-v3-turbo.
    No OpenAI key required — uses existing GROQ_API_KEY.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Detect format from filename if not specified
        if file.filename:
            ext = file.filename.rsplit(".", 1)[-1].lower()
            if ext in {"webm", "mp4", "wav", "mp3", "m4a", "ogg", "flac"}:
                audio_format = ext

        transcribed_text = await audio_processor.transcribe_audio(audio_bytes, audio_format)

        return {
            "status": "success",
            "transcribed_text": transcribed_text,
            "audio_format": audio_format,
            "model": settings.GROQ_WHISPER_MODEL
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.post("/audio/synthesize")
async def synthesize_speech(request: AudioSynthesizeRequest):
    """Text-to-Speech endpoint using OpenAI TTS (requires OPENAI_API_KEY)."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="High-quality TTS requires OPENAI_API_KEY. Use browser SpeechSynthesis instead."
        )
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        audio_bytes = await audio_processor.synthesize_speech(request.text, request.voice)
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {e}")


@app.post("/audio/agent-voice")
async def agent_voice_interaction(
    file: UploadFile = File(...),
    session_id: str = "default_session",
    user_id: str = "default_user",
    audio_format: str = "webm",
    voice: str = "nova"
):
    """HTTP voice-to-voice pipeline (STT → Agent → TTS)."""
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        user_text = await audio_processor.transcribe_audio(audio_bytes, audio_format)
        print(f"[Audio Agent] Transcribed: {user_text}")

        full_response = ""
        async for token in route_and_stream(user_text, session_id, user_id):
            full_response += token

        if not full_response:
            raise HTTPException(status_code=500, detail="Failed to generate response")

        if settings.OPENAI_API_KEY:
            response_audio = await audio_processor.synthesize_speech(full_response, voice)
            return StreamingResponse(
                iter([response_audio]),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=agent_response.mp3",
                    "X-Agent-Text": full_response[:200]
                }
            )
        else:
            return {"status": "text_only", "agent_text": full_response}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice interaction failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Voice Channel  —  /ws/voice
# ──────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/voice")
async def websocket_voice_channel(websocket: WebSocket, session_id: str = "ws_session"):
    """
    Real-time bi-directional voice channel.

    Client → Server messages (JSON):
        {"type": "audio_chunk", "data": "<base64 PCM bytes>", "format": "webm"}
        {"type": "audio_end"}          — user finished speaking
        {"type": "interrupt"}          — interrupt current AI response
        {"type": "vad_energy", "rms": 342.5}  — client-side VAD reading

    Server → Client messages (JSON):
        {"type": "vad_status", "speaking": true/false}
        {"type": "transcript", "text": "...", "final": true/false}
        {"type": "ai_token", "token": "...", "done": false}
        {"type": "ai_done"}
        {"type": "error", "message": "..."}
        {"type": "interrupted"}
    """
    await websocket.accept()
    session = VoiceSession(websocket, session_id)
    active_voice_sessions[session_id] = session
    print(f"[WS] Voice session opened: {session_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ── Audio chunk accumulation ──────────────────────────────────────
            if msg_type == "audio_chunk":
                import base64
                chunk_b64 = msg.get("data", "")
                if chunk_b64:
                    chunk_bytes = base64.b64decode(chunk_b64)
                    session.audio_chunks.append(chunk_bytes)

                # If AI is currently streaming and user sends audio → interrupt
                if session.ai_streaming:
                    session.interrupted = True
                    await websocket.send_text(json.dumps({"type": "interrupted"}))

            # ── VAD energy reading from client ────────────────────────────────
            elif msg_type == "vad_energy":
                rms = float(msg.get("rms", 0))
                speaking = audio_processor.is_speech(rms)
                if speaking != session.is_speaking:
                    session.is_speaking = speaking
                    await websocket.send_text(json.dumps({
                        "type": "vad_status",
                        "speaking": speaking,
                        "rms": rms
                    }))

            # ── User finished speaking — transcribe + respond ─────────────────
            elif msg_type == "audio_end":
                if not session.audio_chunks:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "No audio received"
                    }))
                    continue

                # Transcribe
                audio_format = msg.get("format", "webm")
                try:
                    transcript = await audio_processor.transcribe_chunks(
                        session.audio_chunks, audio_format
                    )
                    session.audio_chunks = []  # reset buffer

                    await websocket.send_text(json.dumps({
                        "type": "transcript",
                        "text": transcript,
                        "final": True
                    }))
                except Exception as exc:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Transcription failed: {exc}"
                    }))
                    session.audio_chunks = []
                    continue

                # Stream AI response
                session.ai_streaming = True
                session.interrupted = False
                try:
                    async for token in route_and_stream(transcript, session_id, "ws_user"):
                        if session.interrupted:
                            break
                        await websocket.send_text(json.dumps({
                            "type": "ai_token",
                            "token": token,
                            "done": False
                        }))
                except Exception as exc:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Agent error: {exc}"
                    }))
                finally:
                    session.ai_streaming = False
                    await websocket.send_text(json.dumps({"type": "ai_done"}))

            # ── Interrupt signal ──────────────────────────────────────────────
            elif msg_type == "interrupt":
                session.interrupted = True
                session.audio_chunks = []
                await websocket.send_text(json.dumps({"type": "interrupted"}))

    except WebSocketDisconnect:
        print(f"[WS] Session disconnected: {session_id}")
    except Exception as exc:
        print(f"[WS] Session error: {exc}")
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        active_voice_sessions.pop(session_id, None)
        print(f"[WS] Session cleaned up: {session_id}")